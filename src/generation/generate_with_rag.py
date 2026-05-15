from __future__ import annotations

from typing import Any, Callable

from src.embedding.vector_store import retrieve_current_context_by_text, retrieve_next_by_text
from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.planner.jepa_dataset import build_generation_context_text
from src.planner.scene_analyzer import analyze_current_scene, build_analyzed_generation_context
from src.utils.config import AppConfig


def _rag_query_context(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None,
) -> tuple[str, dict[str, Any] | None]:
    if config.generation.use_scene_analyzer:
        analysis = analyze_current_scene(config, client, world, characters, previous_scene, scene_preset=scene_preset)
        return build_analyzed_generation_context(world, characters, previous_scene, analysis, scene_preset=scene_preset), analysis
    return build_generation_context_text(world, characters, previous_scene, scene_preset=scene_preset), None


def plan_rag_generation(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None = None,
) -> dict[str, Any]:
    query_context, analysis = _rag_query_context(config, client, world, characters, previous_scene, scene_preset)
    current_retrieved = retrieve_current_context_by_text(config, client, query_context, config.generation.top_k)
    next_retrieved = retrieve_next_by_text(config, client, query_context, config.generation.top_k)
    examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in current_retrieved[: config.generation.rag_context_limit]]
    return {
        "query_context": query_context,
        "analyzed_scene": analysis,
        "retrieved": current_retrieved,
        "current_retrieved": current_retrieved,
        "next_retrieved": next_retrieved,
        "examples": examples,
        "direction": "검색된 유사 장면의 전환 논리를 참고해 다음 갈등을 확장한다.",
    }


def generate_with_rag(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callback: Callable[[str], None] | None = None,
    scene_preset: dict[str, str] | None = None,
    return_details: bool = False,
) -> str | dict[str, Any]:
    plan = plan_rag_generation(config, client, world, characters, previous_scene, scene_preset=scene_preset)
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        direction=plan["direction"],
        examples=plan["examples"],
        beat_card=build_beat_card(
            "RAG + LLM",
            plan["direction"],
            plan["examples"],
            characters,
            config.generation.rag_context_limit,
            scene_preset=scene_preset,
        ),
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
        return {"text": repaired, "rag": plan}
    return repaired
