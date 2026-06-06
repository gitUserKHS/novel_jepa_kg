from __future__ import annotations

import io
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.generation.consistency import (
    allowed_name_instruction,
    extract_character_names,
    repair_name_consistency,
)
from src.generation.generate_with_jepa import TraceCallback, _emit, plan_jepa_generation
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.memory.story_rag import (
    StoryMemory,
    StoryMemoryStreamFilter,
    format_hierarchical_story_context,
    load_story_memories,
    retrieve_story_memories,
    split_story_memory,
    story_memory_instruction,
    write_story_ledger,
    write_story_memories,
)
from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path


CREATIVE_HALLUCINATION_MODE = "Creative Hallucination + JEPA"

NARRATIVE_PHASES = [
    "Open with an immediate disturbance and a concrete sensory problem.",
    "Reveal the first clue and make its meaning uncertain.",
    "Force an uneasy alliance or a difficult promise.",
    "Increase external pressure through pursuit, deadline, or exposure.",
    "Offer temporary safety while planting a subtle contradiction.",
    "Reveal a hidden rule of the world or the central mystery.",
    "Fracture trust between the central characters.",
    "Deliver a midpoint reversal that changes the apparent goal.",
    "Show the personal cost of the new direction.",
    "Isolate the protagonist and narrow the available choices.",
    "Connect earlier clues into a dangerous insight.",
    "Present a convincing but incomplete solution.",
    "Demand a sacrifice that changes a relationship.",
    "Stage the main confrontation and irreversible choice.",
    "Resolve the immediate arc while leaving a resonant final hook.",
]

SECTION_HEADING_RE = re.compile(r"(?m)^###\s+")
BUNDLE_DRAFT = "creative_longform_latest.md"
BUNDLE_MEMORY = "creative_longform_memory.jsonl"
BUNDLE_STATE = "creative_longform_state.json"
BUNDLE_CONTEXT = "creative_longform_context.json"
MAX_IMPORT_BYTES = 50 * 1024 * 1024


def build_hallucination_contract(target_ratio: float) -> str:
    target_pct = int(max(0.0, min(1.0, target_ratio)) * 100)
    return f"""
[Controlled hallucination contract]
- Treat hallucination as creative expansion, not factual drift.
- Keep the world rules, known character names, and previous-scene facts intact.
- Intentionally add 2-4 plausible new narrative elements: a sensory detail, symbol, clue, emotional inference, or pressure point.
- Aim for roughly {target_pct}% novel material compared with the previous scene.
- Every new element must be compatible with the JEPA/RAG direction and must create a useful next-scene hook.
- Do not introduce new proper-name characters unless the user already named them.
- Avoid contradictions, sudden retcons, genre-breaking facts, and copied retrieved events.
- Output only polished prose; do not label the new elements.
""".strip()


def hallucination_temperature(config: AppConfig) -> float:
    base = float(config.generation.temperature)
    delta = float(config.generation.hallucination_temperature_delta)
    return max(0.1, min(1.3, base + delta))


def _section_role(section_index: int, planned_sections: int) -> str:
    if section_index <= len(NARRATIVE_PHASES):
        return NARRATIVE_PHASES[section_index - 1]
    if section_index >= planned_sections:
        return "Deepen the aftermath, resolve one emotional thread, and preserve a final hook without repeating prior events."
    return "Advance the unresolved conflict with a new consequence and a meaningful character decision."


def _section_title(text: str, section_index: int) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            return stripped[4:].strip()[:80]
    return f"장면 {section_index}"


def _normalize_section(text: str, section_index: int) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if not any(line.strip().startswith("### ") for line in cleaned.splitlines()):
        cleaned = f"### 장면 {section_index}\n\n{cleaned}"
    return cleaned


def _write_longform_checkpoint(config: AppConfig, sections: list[str]) -> str:
    path = resolve_path(config, config.generation.longform_checkpoint_path)
    ensure_parent(path)
    path.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")
    return str(path)


def _load_longform_sections(config: AppConfig) -> list[str]:
    path = resolve_path(config, config.generation.longform_checkpoint_path)
    if not path.exists():
        return []
    return _sections_from_text(path.read_text(encoding="utf-8"))


def _sections_from_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [_normalize_section(text, 1)]
    return [
        text[match.start() : matches[index + 1].start() if index + 1 < len(matches) else None].strip()
        for index, match in enumerate(matches)
    ]


def _load_run_state(config: AppConfig) -> dict[str, Any]:
    path = resolve_path(config, config.generation.longform_state_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_run_state(
    config: AppConfig,
    *,
    sections: list[str],
    memories: list[StoryMemory],
    turns_completed: int,
    turn_target_chars: int,
    turn_in_progress: int | None = None,
) -> str:
    path = resolve_path(config, config.generation.longform_state_path)
    ensure_parent(path)
    text = "\n\n".join(sections).strip()
    payload = {
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "turns_completed": turns_completed,
        "turn_in_progress": turn_in_progress,
        "turn_target_chars": turn_target_chars,
        "total_chars": len(text),
        "section_count": len(sections),
        "memory_count": len(memories),
        "checkpoint_path": str(resolve_path(config, config.generation.longform_checkpoint_path)),
        "memory_path": str(resolve_path(config, config.generation.story_memory_path)),
        "ledger_path": str(resolve_path(config, config.generation.story_ledger_path)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def longform_artifact_status(config: AppConfig) -> dict[str, Any]:
    sections = _load_longform_sections(config)
    memories = load_story_memories(resolve_path(config, config.generation.story_memory_path))
    state = _load_run_state(config)
    text = "\n\n".join(sections).strip()
    return {
        "exists": bool(sections),
        "total_chars": len(text),
        "section_count": len(sections),
        "memory_count": len(memories),
        "turns_completed": int(state.get("turns_completed", 0) or 0),
        "checkpoint_path": str(resolve_path(config, config.generation.longform_checkpoint_path)),
        "memory_path": str(resolve_path(config, config.generation.story_memory_path)),
        "ledger_path": str(resolve_path(config, config.generation.story_ledger_path)),
        "state_path": str(resolve_path(config, config.generation.longform_state_path)),
    }


def export_longform_bundle(
    config: AppConfig,
    *,
    world: str = "",
    characters: str = "",
    previous_scene: str = "",
) -> bytes:
    status = longform_artifact_status(config)
    if not status["exists"]:
        raise FileNotFoundError("No saved long-form draft exists to export.")

    context = {
        "version": 1,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "world": world,
        "characters": characters,
        "previous_scene": previous_scene,
        "chat_model": config.ollama.chat_model,
        "context_length": config.ollama.num_ctx,
    }
    sources = {
        BUNDLE_DRAFT: resolve_path(config, config.generation.longform_checkpoint_path),
        BUNDLE_MEMORY: resolve_path(config, config.generation.story_memory_path),
        BUNDLE_STATE: resolve_path(config, config.generation.longform_state_path),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, path in sources.items():
            if path.exists():
                archive.writestr(archive_name, path.read_bytes())
        archive.writestr(
            BUNDLE_CONTEXT,
            json.dumps(context, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return buffer.getvalue()


def import_longform_file(
    config: AppConfig,
    file_name: str,
    content: bytes,
    *,
    characters: str = "",
) -> dict[str, Any]:
    if not content:
        raise ValueError("The selected file is empty.")
    if len(content) > MAX_IMPORT_BYTES:
        raise ValueError("The selected file exceeds the 50 MB import limit.")

    suffix = Path(file_name).suffix.lower()
    context: dict[str, Any] = {}
    imported_state: dict[str, Any] = {}
    memory_text = ""
    if suffix == ".zip":
        draft_text, memory_text, imported_state, context = _read_longform_bundle(content)
    elif suffix in {".md", ".txt"}:
        draft_text = _decode_utf8(content, file_name)
    else:
        raise ValueError("Supported saved files are .md, .txt, and .zip.")

    sections = _sections_from_text(draft_text)
    if not sections:
        raise ValueError("The saved file contains no novel prose.")

    effective_characters = str(context.get("characters") or characters)
    known_names = extract_character_names(effective_characters)
    memories: list[StoryMemory] = []
    if memory_text:
        memory_path = resolve_path(config, config.generation.story_memory_path)
        ensure_parent(memory_path)
        memory_path.write_text(memory_text, encoding="utf-8")
        memories = load_story_memories(memory_path)
    memory_rebuilt = len(memories) != len(sections)
    if memory_rebuilt:
        memories = [
            split_story_memory(section, index, known_names)[1]
            for index, section in enumerate(sections, start=1)
        ]

    checkpoint_path = _write_longform_checkpoint(config, sections)
    memory_path = write_story_memories(
        resolve_path(config, config.generation.story_memory_path),
        memories,
    )
    ledger_path = write_story_ledger(
        resolve_path(config, config.generation.story_ledger_path),
        memories,
        group_size=config.generation.story_summary_group_size,
    )
    turns_completed = max(
        1,
        int(imported_state.get("turns_completed", 1) or 1),
    )
    turn_target = max(
        1000,
        int(imported_state.get("turn_target_chars", config.generation.turn_target_chars) or config.generation.turn_target_chars),
    )
    state_path = _write_run_state(
        config,
        sections=sections,
        memories=memories,
        turns_completed=turns_completed,
        turn_target_chars=turn_target,
        turn_in_progress=None,
    )
    return {
        **longform_artifact_status(config),
        "file_name": file_name,
        "context": context,
        "checkpoint_path": checkpoint_path,
        "memory_path": memory_path,
        "ledger_path": ledger_path,
        "state_path": state_path,
        "memory_rebuilt": memory_rebuilt,
    }


def _read_longform_bundle(content: bytes) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            total_size = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                base_name = Path(info.filename.replace("\\", "/")).name
                if base_name in {BUNDLE_DRAFT, BUNDLE_MEMORY, BUNDLE_STATE, BUNDLE_CONTEXT}:
                    if base_name in members:
                        raise ValueError(f"Duplicate bundle member: {base_name}")
                    total_size += info.file_size
                    if info.file_size > MAX_IMPORT_BYTES or total_size > MAX_IMPORT_BYTES:
                        raise ValueError("The saved bundle expands beyond the 50 MB import limit.")
                    members[base_name] = info
            if BUNDLE_DRAFT not in members:
                raise ValueError(f"The ZIP bundle is missing {BUNDLE_DRAFT}.")
            draft_text = _decode_utf8(archive.read(members[BUNDLE_DRAFT]), BUNDLE_DRAFT)
            memory_text = (
                _decode_utf8(archive.read(members[BUNDLE_MEMORY]), BUNDLE_MEMORY)
                if BUNDLE_MEMORY in members
                else ""
            )
            state = _read_json_member(archive, members.get(BUNDLE_STATE))
            context = _read_json_member(archive, members.get(BUNDLE_CONTEXT))
            return draft_text, memory_text, state, context
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected ZIP is not a valid long-form bundle.") from exc


def _read_json_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo | None,
) -> dict[str, Any]:
    if info is None:
        return {}
    try:
        payload = json.loads(_decode_utf8(archive.read(info), info.filename))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _decode_utf8(content: bytes, file_name: str) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{file_name} must be UTF-8 encoded.") from exc


def generate_with_controlled_hallucination(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callback: Callable[[str], None] | None = None,
    scene_preset: dict[str, str] | None = None,
    return_details: bool = False,
    trace_callback: TraceCallback | None = None,
    continue_existing: bool = False,
    turn_target_chars: int | None = None,
    continuation_instruction: str = "",
) -> str | dict[str, Any]:
    sections = _load_longform_sections(config) if continue_existing else []
    story_memories = (
        load_story_memories(resolve_path(config, config.generation.story_memory_path))
        if continue_existing
        else []
    )
    if continue_existing and not sections:
        raise FileNotFoundError(
            "No long-form checkpoint exists yet. Start a new novel before continuing."
        )
    known_names = extract_character_names(characters)
    if len(story_memories) < len(sections):
        story_memories = [
            split_story_memory(section, index, known_names)[1]
            for index, section in enumerate(sections, start=1)
        ]
    elif len(story_memories) > len(sections):
        story_memories = story_memories[: len(sections)]

    existing_text = "\n\n".join(sections).strip()
    recent_context_chars = max(600, int(config.generation.longform_recent_context_chars))
    recent_excerpt = existing_text[-recent_context_chars:] if existing_text else previous_scene[-recent_context_chars:]
    planner_memory_context, _ledger = format_hierarchical_story_context(
        story_memories,
        [],
        continuation_instruction or recent_excerpt,
        max(800, int(config.generation.story_memory_context_chars) // 2),
        group_size=config.generation.story_summary_group_size,
    )
    planner_previous_scene = "\n\n".join(
        part
        for part in [
            previous_scene.strip()[-1200:],
            "[Compressed prior story state]" if story_memories else "",
            planner_memory_context if story_memories else "",
            "[Recent prose]" if existing_text else "",
            recent_excerpt if existing_text else "",
            "[User continuation instruction]" if continuation_instruction.strip() else "",
            continuation_instruction.strip(),
        ]
        if part
    )
    plan = plan_jepa_generation(
        config,
        client,
        world,
        characters,
        planner_previous_scene,
        scene_preset=scene_preset,
        trace_callback=trace_callback,
    )
    contract = build_hallucination_contract(config.generation.hallucination_target)
    creative_beat_card = "\n".join([str(plan["beat_card"]), contract])
    overall_target_chars = max(1000, int(config.generation.target_novel_chars))
    turn_target = max(1000, int(turn_target_chars or config.generation.turn_target_chars))
    planned_sections = max(1, int(config.generation.section_count))
    max_total_sections = max(planned_sections, int(config.generation.longform_max_sections))
    turn_section_cap = max(1, int(config.generation.turn_max_sections))
    if client.dry_run:
        turn_target = min(turn_target, 2400)
        planned_sections = min(planned_sections, 2)
        turn_section_cap = min(turn_section_cap, 3)
    if len(sections) >= max_total_sections:
        raise RuntimeError(
            f"The draft already reached the configured maximum of {max_total_sections} sections."
        )
    turn_floor = int(turn_target * 0.9)
    section_target_chars = max(600, int(config.generation.section_min_chars))
    expected_turn_sections = max(1, math.ceil(turn_target / section_target_chars))
    turn_section_cap = min(
        turn_section_cap,
        max_total_sections - len(sections),
        max(expected_turn_sections, min(turn_section_cap, expected_turn_sections + 2)),
    )
    run_state = _load_run_state(config) if continue_existing else {}
    turns_completed = int(run_state.get("turns_completed", 0) or 0)
    current_turn = turns_completed + 1
    _emit(
        trace_callback,
        "Add controlled hallucination contract",
        "done",
        {
            "target": config.generation.hallucination_target,
            "temperature": hallucination_temperature(config),
            "overall_target_chars": overall_target_chars,
            "turn_target_chars": turn_target,
            "turn": current_turn,
            "continue_existing": continue_existing,
            "planned_sections": planned_sections,
        },
    )

    section_titles = [_section_title(section, index) for index, section in enumerate(sections, start=1)]
    memory_retrieval_count = 0
    turn_sections: list[str] = []
    turn_start_chars = len(existing_text)
    turn_added_chars = 0
    total_chars = turn_start_chars
    checkpoint_path = _write_longform_checkpoint(config, sections)
    memory_path = write_story_memories(
        resolve_path(config, config.generation.story_memory_path),
        story_memories,
    )
    ledger_path = write_story_ledger(
        resolve_path(config, config.generation.story_ledger_path),
        story_memories,
        group_size=config.generation.story_summary_group_size,
    )
    state_path = _write_run_state(
        config,
        sections=sections,
        memories=story_memories,
        turns_completed=turns_completed,
        turn_target_chars=turn_target,
        turn_in_progress=current_turn,
    )
    section_index = len(sections) + 1
    generated_this_turn = 0
    while (
        turn_added_chars < turn_floor
        and generated_this_turn < turn_section_cap
        and section_index <= max_total_sections
    ):
        recent_excerpt = sections[-1][-recent_context_chars:] if sections else previous_scene[-recent_context_chars:]
        prior_titles = " / ".join(section_titles[-8:]) or "(none)"
        memory_query = "\n".join(
            [
                str(plan["direction"]),
                _section_role(section_index, planned_sections),
                continuation_instruction,
                recent_excerpt,
            ]
        )
        retrieved_memories = (
            retrieve_story_memories(
                story_memories,
                memory_query,
                config.generation.story_memory_top_k,
                section_index,
            )
            if config.generation.enable_story_memory_rag
            else []
        )
        memory_retrieval_count += len(retrieved_memories)
        memory_context, ledger = format_hierarchical_story_context(
            story_memories,
            retrieved_memories,
            memory_query,
            max(400, int(config.generation.story_memory_context_chars)),
            group_size=config.generation.story_summary_group_size,
        )
        projected_total = total_chars + section_target_chars
        nearing_overall_end = projected_total >= int(overall_target_chars * 0.9)
        final_section_of_turn = generated_this_turn + 1 >= expected_turn_sections
        completion_rule = (
            "Move toward climax and resolve the main arc while preserving a resonant final image."
            if nearing_overall_end
            else (
                "End this turn with a strong continuation hook, but do not abruptly cut a sentence or action."
                if final_section_of_turn
                else "Do not resolve the central conflict yet; end with forward pressure."
            )
        )
        section_card = "\n".join(
            [
                creative_beat_card,
                "[Long-form section plan]",
                f"- overall section: {section_index}/{planned_sections}",
                f"- generation turn: {current_turn}",
                f"- narrative role: {_section_role(section_index, planned_sections)}",
                f"- target body length: about {section_target_chars} Korean characters",
                f"- prior section titles: {prior_titles}",
                f"- completion rule: {completion_rule}",
                "[Compressed state, KG, and retrieved memory]",
                memory_context,
                "- Treat retrieved story memory as established canon.",
                "- If two memories differ, the higher section number is the newer state and takes precedence.",
                "- Preserve character, relationship, object, location, clue, and promise states.",
                "- Do not repeat a resolved clue as unresolved or undo a state change without an explicit cause.",
                "- Use only memories relevant to this section; do not recap the ledger.",
                "- Write exactly one section with one `###` Korean subtitle.",
                "- Continue directly from the recent excerpt without recapping the whole story.",
                "- Avoid repeating a subtitle, revelation, confrontation, or emotional beat.",
                (
                    f"- User continuation request: {continuation_instruction.strip()}"
                    if continuation_instruction.strip()
                    else "- Follow the existing unresolved pressure and character goals."
                ),
            ]
        )
        continuity_context = "\n\n".join(
            part
            for part in [
                previous_scene.strip()[-1200:],
                "[Recent generated excerpt]",
                recent_excerpt.strip(),
            ]
            if part
        ).strip()
        prompt = prose_prompt(
            world,
            characters,
            continuity_context,
            config.generation.style,
            direction=plan["direction"],
            examples=plan["examples"],
            beat_card=section_card,
            consistency_rules=allowed_name_instruction(characters),
            sectioned_output=True,
            section_count=1,
            section_min_chars=section_target_chars,
        )
        if config.generation.enable_story_memory_rag:
            prompt += story_memory_instruction(section_index)
        _emit(
            trace_callback,
            "Generate long-form sections",
            "running",
            {
                "section": f"{section_index}/{planned_sections}",
                "turn": current_turn,
                "turn_section": generated_this_turn + 1,
                "turn_chars": turn_added_chars,
                "turn_target_chars": turn_target,
                "current_chars": total_chars,
                "overall_target_chars": overall_target_chars,
                "chunk_tokens": config.generation.max_tokens,
                "story_memory_hits": len(retrieved_memories),
                "active_states": len(ledger.get("current_states", [])),
                "kg_relations": len(ledger.get("relations", [])),
            },
        )
        if stream_callback is not None and turn_sections:
            stream_callback("\n\n")
        try:
            memory_stream_filter = (
                StoryMemoryStreamFilter(stream_callback)
                if stream_callback is not None and config.generation.enable_story_memory_rag
                else None
            )
            raw_section = client.chat(
                prompt,
                system=(
                    "You write long-form Korean novel prose in sequential sections. "
                    "Use controlled creative hallucination while preserving continuity."
                ),
                temperature=hallucination_temperature(config),
                max_tokens=config.generation.max_tokens,
                stream_callback=memory_stream_filter.feed if memory_stream_filter else stream_callback,
            )
            if memory_stream_filter is not None:
                memory_stream_filter.finish()
            section_text, story_memory = split_story_memory(
                raw_section,
                section_index,
                known_names,
            )
            section = _normalize_section(section_text, section_index)
            if config.generation.enable_consistency_repair:
                section = repair_name_consistency(
                    config,
                    client,
                    section,
                    world,
                    characters,
                    continuity_context,
                )
        except Exception as exc:
            checkpoint_path = _write_longform_checkpoint(config, sections)
            memory_path = write_story_memories(
                resolve_path(config, config.generation.story_memory_path),
                story_memories,
            )
            ledger_path = write_story_ledger(
                resolve_path(config, config.generation.story_ledger_path),
                story_memories,
                group_size=config.generation.story_summary_group_size,
            )
            state_path = _write_run_state(
                config,
                sections=sections,
                memories=story_memories,
                turns_completed=turns_completed,
                turn_target_chars=turn_target,
                turn_in_progress=current_turn,
            )
            raise RuntimeError(
                f"Long-form generation stopped at section {section_index}. "
                f"Partial draft saved to {checkpoint_path}; memory saved to {memory_path}; "
                f"ledger saved to {ledger_path}; state saved to {state_path}. {exc}"
            ) from exc
        if not section:
            break
        sections.append(section)
        turn_sections.append(section)
        section_titles.append(_section_title(section, section_index))
        story_memory.title = story_memory.title or section_titles[-1]
        story_memories.append(story_memory)
        total_chars = len("\n\n".join(sections))
        turn_added_chars = total_chars - turn_start_chars
        generated_this_turn += 1
        checkpoint_path = _write_longform_checkpoint(config, sections)
        memory_path = write_story_memories(
            resolve_path(config, config.generation.story_memory_path),
            story_memories,
        )
        ledger_path = write_story_ledger(
            resolve_path(config, config.generation.story_ledger_path),
            story_memories,
            group_size=config.generation.story_summary_group_size,
        )
        state_path = _write_run_state(
            config,
            sections=sections,
            memories=story_memories,
            turns_completed=turns_completed,
            turn_target_chars=turn_target,
            turn_in_progress=current_turn,
        )
        _emit(
            trace_callback,
            "Generate long-form sections",
            "running" if turn_added_chars < turn_floor else "done",
            {
                "completed_sections": len(sections),
                "current_chars": total_chars,
                "turn_chars": turn_added_chars,
                "turn_target_chars": turn_target,
                "overall_target_chars": overall_target_chars,
                "checkpoint": checkpoint_path,
                "story_memories": len(story_memories),
                "story_memory_path": memory_path,
                "story_ledger_path": ledger_path,
                "run_state_path": state_path,
            },
        )
        section_index += 1

    repaired = "\n\n".join(sections).strip()
    turn_text = "\n\n".join(turn_sections).strip()
    turns_completed = current_turn if turn_sections else turns_completed
    state_path = _write_run_state(
        config,
        sections=sections,
        memories=story_memories,
        turns_completed=turns_completed,
        turn_target_chars=turn_target,
        turn_in_progress=None,
    )
    final_ledger = write_story_ledger(
        resolve_path(config, config.generation.story_ledger_path),
        story_memories,
        group_size=config.generation.story_summary_group_size,
    )
    _emit(
        trace_callback,
        "Generate long-form sections",
        "done",
        {
            "completed_sections": len(sections),
            "final_chars": len(repaired),
            "turn": current_turn,
            "turn_chars": len(turn_text),
            "turn_target_chars": turn_target,
            "overall_target_chars": overall_target_chars,
            "checkpoint": checkpoint_path,
            "story_memories": len(story_memories),
            "story_memory_retrievals": memory_retrieval_count,
            "story_memory_path": memory_path,
            "story_ledger_path": final_ledger,
            "run_state_path": state_path,
        },
    )
    if return_details:
        details = {key: value for key, value in plan.items() if key != "predicted_embedding"}
        details["hallucination_contract"] = contract
        details["hallucination_target"] = config.generation.hallucination_target
        details["target_novel_chars"] = overall_target_chars
        details["actual_novel_chars"] = len(repaired)
        details["completed_sections"] = len(sections)
        details["turn_number"] = current_turn
        details["turn_target_chars"] = turn_target
        details["turn_actual_chars"] = len(turn_text)
        details["turn_sections"] = len(turn_sections)
        details["continued_existing"] = continue_existing
        details["checkpoint_path"] = checkpoint_path
        details["story_memory_rag_enabled"] = config.generation.enable_story_memory_rag
        details["story_memory_count"] = len(story_memories)
        details["story_memory_retrievals"] = memory_retrieval_count
        details["story_memory_path"] = memory_path
        details["story_ledger_path"] = final_ledger
        details["run_state_path"] = state_path
        return {"text": repaired, "turn_text": turn_text, "planner": details}
    return repaired
