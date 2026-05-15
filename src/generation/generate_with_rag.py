from __future__ import annotations

from typing import Any, Callable

from src.embedding.vector_store import retrieve_by_text
from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.utils.config import AppConfig


def plan_rag_generation(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
) -> dict[str, Any]:
    retrieved = retrieve_by_text(config, client, previous_scene, config.generation.top_k)
    examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in retrieved[: config.generation.rag_context_limit]]
    return {
        "retrieved": retrieved,
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
    plan = plan_rag_generation(config, client, previous_scene)
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
