from __future__ import annotations

import json
import math
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from src.service.consumer_store import JOB_FAILED_RECOVERABLE, JOB_SUCCEEDED, ConsumerStore
from src.service.story_workspace import (
    StoryWorkspace,
    build_continuation_bundle,
    read_draft,
    split_sections,
)
from src.service.worker import ConsumerWorker
from src.utils.config import AppConfig


def fake_manifest(_config: AppConfig) -> dict[str, Any]:
    return {
        "version": "test-active-v1",
        "paths": {
            "dataset": "unused-dataset.jsonl",
            "checkpoint": "unused-predictor.pt",
            "embeddings": "unused-scenes.npz",
            "current_index": "unused-current.faiss",
            "next_index": "unused-next.faiss",
        },
    }


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_memory_on_continuation = False

    def __call__(
        self,
        config: AppConfig,
        _client: object,
        _world: str,
        _characters: str,
        _previous_scene: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        draft_path = Path(config.generation.longform_checkpoint_path)
        memory_path = Path(config.generation.story_memory_path)
        if kwargs.get("continue_existing"):
            self.saw_memory_on_continuation = memory_path.exists()
        existing = draft_path.read_text(encoding="utf-8") if draft_path.exists() else ""
        target = int(kwargs.get("turn_target_chars") or 3000)
        phrase = "단서가 드러나고 선택의 대가로 관계와 목표가 변했다. "
        body = (phrase * math.ceil(target / len(phrase)))[:target]
        section = f"### {self.calls}장 새로운 선택\n\n{body}"
        full = "\n\n".join(value for value in [existing.strip(), section] if value)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(full, encoding="utf-8")
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"section": self.calls, "summary": "선택과 변화"}, ensure_ascii=False) + "\n")
        Path(config.generation.longform_state_path).write_text("{}", encoding="utf-8")
        Path(config.generation.story_ledger_path).write_text("{}", encoding="utf-8")
        return {
            "text": full,
            "turn_text": section,
            "planner": {
                "retrieval_mean_score": 0.77,
                "story_memory_retrievals": max(0, self.calls - 1),
                "turn_repetition_retries": 0,
                "novel_completed": len(full) >= int(config.generation.target_novel_chars),
            },
        }


class ConsumerWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.user = self.store.create_user(
            username="worker_test",
            display_name="Worker Test",
            password="worker-password",
        )
        self.story = self.store.create_story(
            self.user["id"],
            title="3만 자 테스트",
            genre="SF 미스터리",
            premise="기억 도난 사건",
            world="기억이 거래되는 도시",
            protagonist="서윤: 기록 복원가",
            target_chars=30000,
            research_consent=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_multiple_turns_reach_30000_and_keep_story_memory(self) -> None:
        generator = FakeGenerator()
        worker = ConsumerWorker(
            self.config,
            store=self.store,
            worker_id="test-worker",
            client_factory=lambda _config: object(),
            generator=generator,
            manifest_loader=fake_manifest,
        )
        for turn in range(6):
            job = self.store.enqueue_job(
                self.user["id"],
                self.story["id"],
                instruction=f"{turn + 1}번째 전개",
                creativity_profile="balanced",
                requested_chars=5000,
            )
            self.assertTrue(worker.process_one())
            self.assertEqual(self.store.get_job(job["id"])["status"], JOB_SUCCEEDED)

        workspace = StoryWorkspace.for_story(self.config, self.story["id"])
        draft = read_draft(workspace.draft)
        self.assertGreaterEqual(len(draft), 30000)
        self.assertEqual(len(split_sections(draft)), 6)
        self.assertTrue(generator.saw_memory_on_continuation)
        self.assertEqual(len(workspace.memory.read_text(encoding="utf-8").splitlines()), 6)
        metric_rows = self.store.anonymous_metric_rows()
        self.assertEqual(len(metric_rows), 6)
        self.assertIn("useful_hallucination_score", metric_rows[0])
        self.assertIn("jepa_retrieval_score", metric_rows[0])
        final_job = self.store.get_job(job["id"])
        self.assertTrue(json.loads(final_job["metrics_json"])["novel_completed"])
        completed_story = self.store.get_owned_story(self.user["id"], self.story["id"])
        with ZipFile(BytesIO(build_continuation_bundle(workspace, completed_story))) as archive:
            bundled_draft = archive.read("draft.md").decode("utf-8").replace("\r\n", "\n")
            self.assertEqual(bundled_draft, draft)

    def test_partial_section_failure_is_marked_recoverable(self) -> None:
        def partial_generator(config: AppConfig, *_args: object, **_kwargs: object) -> dict[str, Any]:
            path = Path(config.generation.longform_checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("### 1장 중단된 시작\n\n저장된 부분 원고", encoding="utf-8")
            raise RuntimeError("simulated worker interruption")

        worker = ConsumerWorker(
            self.config,
            store=self.store,
            worker_id="partial-worker",
            client_factory=lambda _config: object(),
            generator=partial_generator,
            manifest_loader=fake_manifest,
        )
        job = self.store.enqueue_job(
            self.user["id"],
            self.story["id"],
            instruction="첫 장면",
            creativity_profile="bold",
            requested_chars=3000,
        )
        self.assertTrue(worker.process_one())
        self.assertEqual(self.store.get_job(job["id"])["status"], JOB_FAILED_RECOVERABLE)
        workspace = StoryWorkspace.for_story(self.config, self.story["id"])
        self.assertIn("저장된 부분 원고", read_draft(workspace.draft))


if __name__ == "__main__":
    unittest.main()
