from __future__ import annotations

import math
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
    format_story_memory_context,
    retrieve_story_memories,
    split_story_memory,
    story_memory_instruction,
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
) -> str | dict[str, Any]:
    plan = plan_jepa_generation(
        config,
        client,
        world,
        characters,
        previous_scene,
        scene_preset=scene_preset,
        trace_callback=trace_callback,
    )
    contract = build_hallucination_contract(config.generation.hallucination_target)
    creative_beat_card = "\n".join([str(plan["beat_card"]), contract])
    target_chars = max(1000, int(config.generation.target_novel_chars))
    planned_sections = max(1, int(config.generation.section_count))
    max_sections = max(planned_sections, int(config.generation.longform_max_sections))
    recent_context_chars = max(400, int(config.generation.longform_recent_context_chars))
    if client.dry_run:
        target_chars = min(target_chars, 2400)
        planned_sections = min(planned_sections, 2)
        max_sections = min(max_sections, 3)
    target_floor = int(target_chars * 0.9)
    section_target_chars = max(
        int(config.generation.section_min_chars),
        math.ceil(target_chars / planned_sections),
    )
    _emit(
        trace_callback,
        "Add controlled hallucination contract",
        "done",
        {
            "target": config.generation.hallucination_target,
            "temperature": hallucination_temperature(config),
            "novel_target_chars": target_chars,
            "planned_sections": planned_sections,
        },
    )

    sections: list[str] = []
    section_titles: list[str] = []
    story_memories: list[StoryMemory] = []
    memory_retrieval_count = 0
    total_chars = 0
    checkpoint_path = _write_longform_checkpoint(config, sections)
    memory_path = write_story_memories(
        resolve_path(config, config.generation.story_memory_path),
        story_memories,
    )
    known_names = extract_character_names(characters)
    section_index = 1
    while section_index <= planned_sections or (total_chars < target_floor and section_index <= max_sections):
        recent_excerpt = sections[-1][-recent_context_chars:] if sections else previous_scene[-recent_context_chars:]
        prior_titles = " / ".join(section_titles[-8:]) or "(none)"
        memory_query = "\n".join(
            [
                str(plan["direction"]),
                _section_role(section_index, planned_sections),
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
        memory_context = format_story_memory_context(
            retrieved_memories,
            max(400, int(config.generation.story_memory_context_chars)),
        )
        completion_rule = (
            "Do not resolve the central conflict yet; end with forward pressure."
            if section_index < planned_sections
            else "Move toward climax, aftermath, and a satisfying but open final image."
        )
        section_card = "\n".join(
            [
                creative_beat_card,
                "[Long-form section plan]",
                f"- section: {section_index}/{planned_sections}",
                f"- narrative role: {_section_role(section_index, planned_sections)}",
                f"- target body length: about {section_target_chars} Korean characters",
                f"- prior section titles: {prior_titles}",
                f"- completion rule: {completion_rule}",
                "[Retrieved story memory for continuity]",
                memory_context,
                "- Treat retrieved story memory as established canon.",
                "- If two memories differ, the higher section number is the newer state and takes precedence.",
                "- Preserve character, relationship, object, location, clue, and promise states.",
                "- Do not repeat a resolved clue as unresolved or undo a state change without an explicit cause.",
                "- Use only memories relevant to this section; do not recap the ledger.",
                "- Write exactly one section with one `###` Korean subtitle.",
                "- Continue directly from the recent excerpt without recapping the whole story.",
                "- Avoid repeating a subtitle, revelation, confrontation, or emotional beat.",
            ]
        )
        continuity_context = "\n\n".join(
            [
                previous_scene.strip(),
                "[Recent generated excerpt]",
                recent_excerpt.strip(),
            ]
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
                "current_chars": total_chars,
                "target_chars": target_chars,
                "chunk_tokens": config.generation.max_tokens,
                "story_memory_hits": len(retrieved_memories),
            },
        )
        if stream_callback is not None and sections:
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
                _, story_memory = split_story_memory(section, section_index, known_names)
        except Exception as exc:
            checkpoint_path = _write_longform_checkpoint(config, sections)
            memory_path = write_story_memories(
                resolve_path(config, config.generation.story_memory_path),
                story_memories,
            )
            raise RuntimeError(
                f"Long-form generation stopped at section {section_index}. "
                f"Partial draft saved to {checkpoint_path}; story memory saved to {memory_path}. {exc}"
            ) from exc
        if not section:
            break
        sections.append(section)
        section_titles.append(_section_title(section, section_index))
        story_memory.title = story_memory.title or section_titles[-1]
        story_memories.append(story_memory)
        total_chars = len("\n\n".join(sections))
        checkpoint_path = _write_longform_checkpoint(config, sections)
        memory_path = write_story_memories(
            resolve_path(config, config.generation.story_memory_path),
            story_memories,
        )
        _emit(
            trace_callback,
            "Generate long-form sections",
            "running" if total_chars < target_floor else "done",
            {
                "completed_sections": len(sections),
                "current_chars": total_chars,
                "target_chars": target_chars,
                "checkpoint": checkpoint_path,
                "story_memories": len(story_memories),
                "story_memory_path": memory_path,
            },
        )
        section_index += 1

    repaired = "\n\n".join(sections).strip()
    _emit(
        trace_callback,
        "Generate long-form sections",
        "done",
        {
            "completed_sections": len(sections),
            "final_chars": len(repaired),
            "target_chars": target_chars,
            "checkpoint": checkpoint_path,
            "story_memories": len(story_memories),
            "story_memory_retrievals": memory_retrieval_count,
            "story_memory_path": memory_path,
        },
    )
    if return_details:
        details = {key: value for key, value in plan.items() if key != "predicted_embedding"}
        details["hallucination_contract"] = contract
        details["hallucination_target"] = config.generation.hallucination_target
        details["target_novel_chars"] = target_chars
        details["actual_novel_chars"] = len(repaired)
        details["completed_sections"] = len(sections)
        details["checkpoint_path"] = checkpoint_path
        details["story_memory_rag_enabled"] = config.generation.enable_story_memory_rag
        details["story_memory_count"] = len(story_memories)
        details["story_memory_retrievals"] = memory_retrieval_count
        details["story_memory_path"] = memory_path
        return {"text": repaired, "planner": details}
    return repaired
