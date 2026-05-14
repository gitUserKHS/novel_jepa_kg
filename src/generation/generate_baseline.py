from __future__ import annotations

from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.utils.config import AppConfig


def generate_llm_only(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
) -> str:
    prompt = prose_prompt(world, characters, previous_scene, config.generation.style)
    return client.chat(prompt, system="당신은 한국어 장편 웹소설 작가입니다.", temperature=config.generation.temperature, max_tokens=config.generation.max_tokens)
