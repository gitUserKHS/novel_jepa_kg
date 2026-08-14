from __future__ import annotations

import re
import shutil
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path


STORY_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SECTION_HEADING_RE = re.compile(r"(?m)^###\s+")


@dataclass(frozen=True)
class StoryWorkspace:
    root: Path
    draft: Path
    state: Path
    memory: Path
    ledger: Path
    outline: Path
    # Prose for the section being written right now. The worker runs in its own
    # process, so a stream callback cannot reach the browser directly; this file
    # is the handoff the consumer UI polls.
    live: Path

    @classmethod
    def for_story(
        cls,
        config: AppConfig,
        story_id: str,
        *,
        create: bool = False,
    ) -> "StoryWorkspace":
        root = safe_story_path(config, story_id)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            draft=root / "draft.md",
            state=root / "state.json",
            memory=root / "memory.jsonl",
            ledger=root / "ledger.json",
            outline=root / "outline.json",
            live=root / "live.txt",
        )

    def delete(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def reset(self) -> None:
        """Erase the manuscript and everything derived from it.

        The directory itself stays, so a reset story can be written again under
        the same id. Every artifact a later section would read has to go
        together: keeping a stale memory, ledger, or outline beside an empty
        draft would make the next section continue a story that no longer exists.
        """
        for path in (self.draft, self.state, self.memory, self.ledger, self.outline, self.live):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def safe_story_path(config: AppConfig, story_id: str) -> Path:
    if not STORY_ID_RE.fullmatch(story_id):
        raise ValueError("Invalid story id.")
    base = resolve_path(config, config.consumer.story_root).resolve()
    candidate = (base / story_id).resolve()
    if candidate.parent != base:
        raise ValueError("Story path escaped the configured story root.")
    return candidate


def configure_story_run(
    config: AppConfig,
    story_id: str,
    *,
    target_chars: int,
    turn_chars: int,
    creativity: float,
    active_manifest: dict[str, Any],
    genre: str = "",
) -> tuple[AppConfig, StoryWorkspace]:
    run_config = config.model_copy(deep=True)
    workspace = StoryWorkspace.for_story(run_config, story_id, create=True)
    run_config.generation.genre = genre
    run_config.ollama.chat_model = run_config.consumer.chat_model
    run_config.ollama.embed_model = run_config.consumer.embed_model
    run_config.generation.target_novel_chars = target_chars
    run_config.generation.turn_target_chars = turn_chars
    run_config.generation.hallucination_target = creativity
    run_config.generation.longform_checkpoint_path = str(workspace.draft)
    run_config.generation.longform_state_path = str(workspace.state)
    run_config.generation.story_memory_path = str(workspace.memory)
    run_config.generation.story_ledger_path = str(workspace.ledger)
    run_config.generation.story_outline_path = str(workspace.outline)

    paths = active_manifest.get("paths", {})
    required = {
        "dataset": "data.filtered_path",
        "checkpoint": "training.checkpoint_path",
        "embeddings": "data.embeddings_path",
        "current_index": "data.current_context_index_path",
        "next_index": "data.faiss_index_path",
    }
    for key in required:
        value = paths.get(key)
        if not value:
            raise ValueError(f"Active JEPA manifest is missing paths.{key}.")
    run_config.training.checkpoint_path = str(paths["checkpoint"])
    run_config.data.filtered_path = str(paths["dataset"])
    run_config.data.embeddings_path = str(paths["embeddings"])
    run_config.data.current_context_index_path = str(paths["current_index"])
    run_config.data.faiss_index_path = str(paths["next_index"])
    return run_config, workspace


def read_draft(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_live_prose(workspace: StoryWorkspace) -> str:
    """Prose of the in-flight section, or "" when nothing is being written.

    Reading races with the worker's appends by design: a torn read just shows a
    slightly shorter tail on one poll, which the next poll corrects.
    """
    try:
        return workspace.live.read_text(encoding="utf-8") if workspace.live.exists() else ""
    except OSError:
        return ""


def clear_live_prose(workspace: StoryWorkspace) -> None:
    try:
        workspace.live.unlink(missing_ok=True)
    except OSError:
        pass


class LiveProseWriter:
    """Append streamed prose to the workspace live file for the consumer UI.

    Flushes are batched so a per-character callback does not turn into one
    filesystem write per character.
    """

    def __init__(self, workspace: StoryWorkspace, *, flush_chars: int = 48) -> None:
        self.workspace = workspace
        self.flush_chars = max(1, flush_chars)
        self._pending: list[str] = []
        self._pending_chars = 0

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._pending.append(chunk)
        self._pending_chars += len(chunk)
        if self._pending_chars >= self.flush_chars:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        text = "".join(self._pending)
        self._pending.clear()
        self._pending_chars = 0
        try:
            ensure_parent(self.workspace.live)
            with self.workspace.live.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            # The live view is cosmetic; never fail a generation over it.
            pass

    def reset(self) -> None:
        self._pending.clear()
        self._pending_chars = 0
        clear_live_prose(self.workspace)


def split_sections(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    starts = [match.start() for match in SECTION_HEADING_RE.finditer(normalized)]
    if not starts:
        return [normalized]
    sections: list[str] = []
    if starts[0] > 0 and normalized[: starts[0]].strip():
        sections.append(normalized[: starts[0]].strip())
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
        section = normalized[start:end].strip()
        if section:
            sections.append(section)
    return sections


def draft_progress(workspace: StoryWorkspace) -> tuple[int, int]:
    text = read_draft(workspace.draft)
    return len(text), len(split_sections(text))


def build_continuation_bundle(workspace: StoryWorkspace, story: dict[str, Any]) -> bytes:
    public_story = {
        key: value
        for key, value in story.items()
        if key not in {"key_salt", "key_hash", "deleted_at"}
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in (
            (workspace.draft, "draft.md"),
            (workspace.state, "state.json"),
            (workspace.memory, "memory.jsonl"),
            (workspace.ledger, "ledger.json"),
            (workspace.outline, "outline.json"),
        ):
            if source.exists():
                archive.writestr(name, source.read_bytes())
        archive.writestr(
            "story.json",
            json.dumps(public_story, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return buffer.getvalue()
