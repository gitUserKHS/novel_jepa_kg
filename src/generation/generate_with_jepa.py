from __future__ import annotations

from typing import Any, Callable

from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.planner.predict import predict_next_embedding_with_diagnostics
from src.utils.config import AppConfig


def _first_transition(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieved:
        return {}
    return retrieved[0].get("sample", {}).get("scene_t_plus_1", {})


def _next_hook(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return ""
    metadata = retrieved[0].get("sample", {}).get("metadata", {})
    plan = metadata.get("diversity_plan") or {}
    return str(plan.get("next_hook", ""))


def build_jepa_beat_card(
    direction: str,
    retrieved: list[dict[str, Any]],
    characters: str,
    context_limit: int,
    scene_preset: dict[str, str] | None = None,
) -> str:
    transition = _first_transition(retrieved)
    evidence = [item.get("sample", {}).get("scene_t_plus_1", {}).get("summary", "") for item in retrieved[:context_limit]]
    state = transition.get("state", [])
    if isinstance(state, list):
        pressure = " / ".join(str(item) for item in state[:3]) or "새 단서 또는 선택 압박"
    else:
        pressure = str(state or "새 단서 또는 선택 압박")
    base = build_beat_card(
        "JEPA-inspired Planner + RAG + LLM",
        direction,
        [item for item in evidence if item][:1],
        characters,
        1,
        scene_preset=scene_preset,
    )
    return "\n".join(
        [
            base,
            "- Planner predicted direction: " + (direction or "이전 장면의 갈등을 한 단계 진전한다."),
            "- Likely conflict movement: " + str(transition.get("conflict", "검색된 전환의 갈등 압박을 압축해 반영한다.")),
            "- Emotional transition: " + str(transition.get("emotion", "감정 상태를 한 단계 변화시킨다.")),
            "- New clue or pressure: " + pressure,
            "- Next hook: " + (_next_hook(retrieved) or "다음 장면으로 이어질 선택지를 남긴다."),
            "- Forbidden: 입력 인물표에 없는 새 고유명사 인물명을 만들지 않는다.",
        ]
    )


def plan_jepa_generation(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None = None,
) -> dict[str, Any]:
    diagnostics = predict_next_embedding_with_diagnostics(
        config,
        client,
        previous_scene,
        world=world,
        characters=characters,
        scene_preset=scene_preset,
    )
    retrieved = diagnostics["retrieved"]
    transition = _first_transition(retrieved)
    direction = str(transition.get("summary") or "이전 장면의 갈등을 한 단계 진전시킨다.")
    beat_card = build_jepa_beat_card(
        direction,
        retrieved,
        characters,
        config.generation.rag_context_limit,
        scene_preset=scene_preset,
    )
    return {
        **diagnostics,
        "direction": direction,
        "beat_card": beat_card,
        "examples": [transition.get("summary", "")] if transition.get("summary") else [],
    }


def generate_with_jepa(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callback: Callable[[str], None] | None = None,
    scene_preset: dict[str, str] | None = None,
    return_details: bool = False,
) -> str | dict[str, Any]:
    plan = plan_jepa_generation(config, client, world, characters, previous_scene, scene_preset=scene_preset)
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        direction=plan["direction"],
        examples=plan["examples"],
        beat_card=plan["beat_card"],
        consistency_rules=allowed_name_instruction(characters),
    )
    text = client.chat(
        prompt,
        system="당신은 한국어 장편 웹소설 작가입니다.",
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        stream_callback=stream_callback,
    )
    repaired = repair_name_consistency(config, client, text, world, characters, previous_scene)
    if return_details:
        return {"text": repaired, "planner": {key: value for key, value in plan.items() if key != "predicted_embedding"}}
    return repaired
