from __future__ import annotations

from src.embedding.vector_store import retrieve_by_text
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.utils.config import AppConfig


def generate_with_rag(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
) -> str:
    retrieved = retrieve_by_text(config, client, previous_scene, config.generation.top_k)
    examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in retrieved]
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        direction="검색된 유사 장면의 전환 논리를 참고해 다음 갈등을 확장한다.",
        examples=examples,
    )
    return client.chat(prompt, system="당신은 한국어 장편 웹소설 작가입니다.", temperature=config.generation.temperature, max_tokens=config.generation.max_tokens)
