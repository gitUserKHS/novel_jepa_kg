from __future__ import annotations

from typing import Callable

from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.utils.config import AppConfig


def generate_llm_only(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    prompt = prose_prompt(
        world,
        characters,
        previous_scene,
        config.generation.style,
        beat_card=build_beat_card("LLM only", None, [], characters, config.generation.rag_context_limit),
        consistency_rules=allowed_name_instruction(characters),
    )
    text = client.chat(
        prompt,
        system="당신은 한국어 장편 웹소설 작가입니다.",
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        stream_callback=stream_callback,
    )
    return repair_name_consistency(config, client, text, world, characters, previous_scene)
