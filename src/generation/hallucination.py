from __future__ import annotations

from typing import Any, Callable

from src.generation.consistency import allowed_name_instruction, repair_name_consistency
from src.generation.generate_with_jepa import TraceCallback, _emit, plan_jepa_generation
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.utils.config import AppConfig


CREATIVE_HALLUCINATION_MODE = "Controlled Hallucination + JEPA"


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
    _emit(
        trace_callback,
        "Add controlled hallucination contract",
        "done",
        {
            "target": config.generation.hallucination_target,
            "temperature": hallucination_temperature(config),
        },
    )
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        direction=plan["direction"],
        examples=plan["examples"],
        beat_card=creative_beat_card,
        consistency_rules=allowed_name_instruction(characters),
        sectioned_output=config.generation.sectioned_output,
        section_count=config.generation.section_count,
        section_min_chars=config.generation.section_min_chars,
    )
    _emit(trace_callback, "Assemble creative prompt", "done", {"prompt_chars": len(prompt)})
    _emit(trace_callback, "Generate creative prose", "running", {"model": config.ollama.chat_model})
    text = client.chat(
        prompt,
        system=(
            "You write Korean novel prose. Use controlled creative hallucination: "
            "add plausible new story material while preserving continuity."
        ),
        temperature=hallucination_temperature(config),
        max_tokens=config.generation.max_tokens,
        stream_callback=stream_callback,
    )
    _emit(trace_callback, "Generate creative prose", "done", {"raw_chars": len(text)})
    _emit(trace_callback, "Consistency repair", "running", {"enabled": config.generation.enable_consistency_repair})
    repaired = repair_name_consistency(config, client, text, world, characters, previous_scene)
    _emit(trace_callback, "Consistency repair", "done", {"final_chars": len(repaired)})
    if return_details:
        details = {key: value for key, value in plan.items() if key != "predicted_embedding"}
        details["hallucination_contract"] = contract
        details["hallucination_target"] = config.generation.hallucination_target
        return {"text": repaired, "planner": details}
    return repaired
