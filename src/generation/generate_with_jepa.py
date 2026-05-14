from __future__ import annotations

from src.embedding.vector_store import retrieve_by_vector
from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.planner.predict import predict_next_embedding
from src.utils.config import AppConfig


def generate_with_jepa(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
) -> str:
    predicted = predict_next_embedding(config, client, previous_scene)
    retrieved = retrieve_by_vector(config, predicted, config.generation.top_k)
    examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in retrieved[: config.generation.rag_context_limit]]
    direction = examples[0] if examples else "이전 장면의 갈등을 한 단계 진전시킨다."
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        direction=direction,
        examples=examples,
        beat_card=build_beat_card(
            "JEPA Planner + RAG + LLM",
            direction,
            examples,
            characters,
            config.generation.rag_context_limit,
        ),
        consistency_rules=allowed_name_instruction(characters),
    )
    text = client.chat(
        prompt,
        system="당신은 한국어 장편 웹소설 작가입니다.",
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )
    return repair_name_consistency(config, client, text, world, characters, previous_scene)
