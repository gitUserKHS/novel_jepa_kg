from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    contradiction_check,
    hallucination_metrics,
    progression_score,
    repetition_rate,
)
from src.generation.consistency import check_name_consistency
from src.generation.hallucination import generate_with_controlled_hallucination
from src.generation.stability import assess_section_stability
from src.llm.ollama_client import OllamaClient
from src.memory.story_rag import (
    StoryMemory,
    build_story_ledger,
    load_story_memories,
    split_story_memory,
)
from src.service.artifacts import ActiveModelUnavailable, load_active_manifest
from src.service.consumer_store import CREATIVITY_LEVELS, MAINTENANCE_ACTIVE, ConsumerStore
from src.service.job_lock import ServiceBusyError, acquire_lock_file, acquire_project_job
from src.service.runtime import make_ollama_client
from src.service.story_workspace import (
    LiveProseWriter,
    StoryWorkspace,
    configure_story_run,
    draft_progress,
    read_draft,
    split_sections,
)
from src.utils.config import AppConfig, load_config
from src.utils.paths import resolve_path


logger = logging.getLogger(__name__)
Generator = Callable[..., str | dict[str, Any]]
ClientFactory = Callable[[AppConfig], OllamaClient]
ManifestLoader = Callable[[AppConfig], dict[str, Any]]


def _public_error(exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:700]
    return f"생성 중 오류가 발생했어. {detail}" if detail else "생성 중 오류가 발생했어."


def _character_sheet(story: dict[str, Any]) -> str:
    explicit = str(story.get("characters", "")).strip()
    protagonist = str(story.get("protagonist", "")).strip()
    if explicit:
        return f"{protagonist}\n{explicit}" if protagonist and protagonist not in explicit else explicit
    if ":" in protagonist or "：" in protagonist:
        return protagonist
    return f"{protagonist}: 작품의 주인공"


def _world_sheet(story: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"장르: {story['genre']}",
            f"핵심 소재: {story['premise']}",
            f"세계관: {story['world']}",
        ]
    )


def _section_metrics(
    sections: list[str],
    start_index: int,
    *,
    characters: str,
    target_creativity: float,
    planner: dict[str, Any],
    memories: list[StoryMemory] | None = None,
    minimum_chars: int = 300,
) -> list[tuple[int, dict[str, Any]]]:
    output: list[tuple[int, dict[str, Any]]] = []
    planner_score = float(planner.get("retrieval_mean_score", 0.0))
    memory_retrievals = int(planner.get("story_memory_retrievals", 0))
    coherence_by_section = {
        int(key): float(value)
        for key, value in dict(planner.get("jepa_coherence_by_section", {})).items()
    }
    metric_memories = list(memories or [])
    while len(metric_memories) < len(sections):
        index = len(metric_memories)
        metric_memories.append(
            split_story_memory(sections[index], index + 1)[1]
        )
    for zero_index in range(start_index, len(sections)):
        section = sections[zero_index]
        previous = sections[zero_index - 1] if zero_index > 0 else ""
        consistency = check_name_consistency(section, characters)
        contradictions = contradiction_check("\n\n".join([previous, section]).strip())
        creative = hallucination_metrics(
            previous,
            section,
            target_novelty=target_creativity,
            contradictions=contradictions,
            name_consistency_score=consistency.score,
        )
        continuity_score = max(0.0, 1.0 - min(1.0, len(contradictions) * 0.25))
        stability = assess_section_stability(
            section,
            metric_memories[zero_index],
            ledger=build_story_ledger(metric_memories[:zero_index]),
            characters=characters,
            prior_titles=[memory.title for memory in metric_memories[:zero_index]],
            minimum_chars=minimum_chars,
        )
        combined = "\n\n".join([previous, section]).strip()
        values: dict[str, Any] = {
            **creative,
            "section_chars": len(section),
            "name_consistency_score": round(consistency.score, 4),
            "name_consistency_issue_count": len(consistency.issues),
            "state_consistency_score": round(continuity_score, 4),
            "time_consistency_score": round(continuity_score, 4),
            "location_consistency_score": round(continuity_score, 4),
            "consistency_issue_count": len(contradictions),
            "new_event_progression_score": round(progression_score(section), 4),
            "narrative_repetition_rate": round(repetition_rate(combined), 4),
            "jepa_retrieval_score": round(planner_score, 4),
            "story_memory_retrievals": memory_retrievals,
            "generation_stability_score": stability.score,
            "generation_stability_issue_count": len(stability.issues),
            "generation_stability_hard_failure": stability.hard_failure,
        }
        # Only sections written this turn were scored by the gate; older rows
        # keep whatever they were given when they were generated.
        if zero_index + 1 in coherence_by_section:
            values["jepa_coherence_score"] = round(coherence_by_section[zero_index + 1], 4)
        output.append((zero_index + 1, values))
    return output


class _Heartbeat:
    def __init__(self, store: ConsumerStore, worker_id: str, job_id: int, interval: float) -> None:
        self.store = store
        self.worker_id = worker_id
        self.job_id = job_id
        self.interval = max(1.0, interval)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="novel-jepa-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.store.heartbeat_job(self.job_id, self.worker_id)
                self.store.heartbeat_worker(self.worker_id, "running")
            except Exception:
                logger.exception("Could not write worker heartbeat")

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 1.0)


class ConsumerWorker:
    def __init__(
        self,
        config: AppConfig,
        *,
        store: ConsumerStore | None = None,
        worker_id: str | None = None,
        client_factory: ClientFactory | None = None,
        generator: Generator = generate_with_controlled_hallucination,
        manifest_loader: ManifestLoader | None = None,
    ) -> None:
        self.config = config
        self.store = store or ConsumerStore(config)
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.client_factory = client_factory or (lambda value: make_ollama_client(value))
        self.generator = generator
        self.manifest_loader = manifest_loader or (
            lambda value: load_active_manifest(value, verify_files=True)
        )

    def process_one(self) -> bool:
        self.store.recover_stale_jobs()
        if self.store.maintenance_status() == MAINTENANCE_ACTIVE:
            self.store.heartbeat_worker(self.worker_id, "maintenance")
            return False
        if self.store.queue_stats()["queued"] == 0:
            self.store.heartbeat_worker(self.worker_id, "idle")
            return False
        try:
            manifest = self.manifest_loader(self.config)
        except ActiveModelUnavailable as exc:
            self.store.heartbeat_worker(self.worker_id, "model_unavailable")
            logger.warning("Consumer worker waiting for active model: %s", exc)
            return False

        try:
            lease = acquire_project_job(self.config, "consumer generation worker")
        except ServiceBusyError:
            self.store.heartbeat_worker(self.worker_id, "waiting_for_gpu")
            return False

        with lease:
            job = self.store.claim_next_job(self.worker_id, str(manifest["version"]))
            if job is None:
                self.store.heartbeat_worker(self.worker_id, "idle")
                return False
            self._run_claimed_job(job, manifest)
            return True

    def _run_claimed_job(self, job: dict[str, Any], manifest: dict[str, Any]) -> None:
        job_id = int(job["id"])
        story_id = str(job["story_id"])
        story = self.store.get_story(story_id)
        if story is None:
            self.store.fail_job(job_id, "작품을 찾을 수 없어 생성하지 못했어.", recoverable=False)
            return

        workspace = StoryWorkspace.for_story(self.config, story_id, create=True)
        before_chars, before_sections = draft_progress(workspace)
        self.store.sync_story_progress(story_id, before_chars, before_sections)
        self.store.set_job_start_section_count(job_id, before_sections)
        characters = _character_sheet(story)
        world = _world_sheet(story)
        previous_sections = split_sections(read_draft(workspace.draft))
        previous_scene = (
            previous_sections[-1]
            if previous_sections
            else f"작품 제목: {story['title']}\n핵심 소재: {story['premise']}\n첫 장면을 시작한다."
        )
        remaining = max(1000, int(story["target_chars"]) - before_chars)
        turn_chars = min(int(job["requested_chars"]), remaining)
        creativity = CREATIVITY_LEVELS[str(job["creativity_profile"])]
        run_config, workspace = configure_story_run(
            self.config,
            story_id,
            target_chars=int(story["target_chars"]),
            turn_chars=turn_chars,
            creativity=creativity,
            active_manifest=manifest,
            genre=str(story["genre"]),
        )
        client = self.client_factory(run_config)
        # Generation runs in this worker process, so the only way the browser can
        # watch prose arrive is through a file the consumer UI polls.
        live = LiveProseWriter(workspace)
        live.reset()

        try:
            with _Heartbeat(
                self.store,
                self.worker_id,
                job_id,
                self.config.consumer.worker_heartbeat_sec,
            ):
                result = self.generator(
                    run_config,
                    client,
                    world,
                    characters,
                    previous_scene,
                    stream_callback=live.feed,
                    return_details=True,
                    continue_existing=before_sections > 0,
                    turn_target_chars=turn_chars,
                    continuation_instruction=str(job["instruction"]),
                )
            if not isinstance(result, dict):
                raise RuntimeError("Long-form generator did not return generation details.")
            after_text = read_draft(workspace.draft)
            after_sections = split_sections(after_text)
            planner = dict(result.get("planner", {}))
            memories = load_story_memories(workspace.memory)
            rows = _section_metrics(
                after_sections,
                before_sections,
                characters=characters,
                target_creativity=creativity,
                planner=planner,
                memories=memories,
                minimum_chars=int(
                    run_config.generation.section_min_chars
                    * run_config.generation.stability_min_section_ratio
                ),
            )
            self.store.save_section_metrics(
                story_id=story_id,
                job_id=job_id,
                model_version=str(manifest["version"]),
                creativity_profile=str(job["creativity_profile"]),
                values=rows,
            )
            job_metrics = {
                "turn_chars": max(0, len(after_text) - before_chars),
                "new_sections": max(0, len(after_sections) - before_sections),
                "jepa_retrieval_score": float(planner.get("retrieval_mean_score", 0.0)),
                "story_memory_retrievals": int(planner.get("story_memory_retrievals", 0)),
                "repetition_retries": int(planner.get("turn_repetition_retries", 0)),
                "stability_retries": int(planner.get("turn_stability_retries", 0)),
                "stability_retry_successes": int(
                    planner.get("turn_stability_retry_successes", 0)
                ),
                "mean_stability_score": float(planner.get("mean_stability_score", 0.0)),
                "mean_jepa_coherence": float(planner.get("mean_jepa_coherence", 0.0)),
                "min_jepa_coherence": float(planner.get("min_jepa_coherence", 0.0)),
                "coherence_retries": int(planner.get("turn_coherence_retries", 0)),
                "coherence_retry_successes": int(
                    planner.get("turn_coherence_retry_successes", 0)
                ),
                "story_outline_beats": int(planner.get("story_outline_beats", 0)),
                "novel_completed": bool(planner.get("novel_completed", False)),
            }
            self.store.complete_job(
                job_id,
                result_chars=max(0, len(after_text) - before_chars),
                result_section_count=max(0, len(after_sections) - before_sections),
                total_chars=len(after_text),
                total_section_count=len(after_sections),
                metrics=job_metrics,
                novel_completed=bool(job_metrics["novel_completed"]),
            )
            self.store.heartbeat_worker(self.worker_id, "idle")
        except Exception as exc:
            logger.exception("Consumer generation job %s failed", job_id)
            after_chars, after_section_count = draft_progress(workspace)
            self.store.sync_story_progress(story_id, after_chars, after_section_count)
            self.store.fail_job(
                job_id,
                _public_error(exc),
                recoverable=after_section_count > before_sections,
            )
            self.store.heartbeat_worker(self.worker_id, "idle")
        finally:
            # Committed sections live in draft.md; leaving the live tail behind
            # would double them in the reader's view.
            live.reset()

    def run_forever(self) -> None:
        logger.info("Consumer worker %s started", self.worker_id)
        last_cleanup = 0.0
        while True:
            self.store.heartbeat_worker(self.worker_id, "polling")
            processed = self.process_one()
            now = time.monotonic()
            if now - last_cleanup >= 3600:
                self.store.purge_expired()
                self.store.purge_expired_user_sessions()
                last_cleanup = now
            if not processed:
                time.sleep(max(0.2, self.config.consumer.worker_poll_sec))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Novel JEPA consumer generation worker")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    worker = ConsumerWorker(config)
    if args.once:
        result = worker.process_one()
        print(json.dumps({"processed": result, "worker_id": worker.worker_id}, ensure_ascii=False))
        return
    # A launcher that is closed without running its cleanup leaves its hidden
    # worker behind. Without this lock those pile up invisibly, and each one
    # holds enough commit charge to make llama-server fail to allocate.
    try:
        lease = acquire_lock_file(
            resolve_path(config, config.consumer.worker_lock_path),
            "consumer worker",
        )
    except ServiceBusyError as exc:
        logger.warning("Another consumer worker is already running; exiting. %s", exc)
        return
    with lease:
        worker.run_forever()


if __name__ == "__main__":
    main()
