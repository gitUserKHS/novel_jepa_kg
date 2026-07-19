from __future__ import annotations

from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig


def make_ollama_client(config: AppConfig, *, dry_run: bool = False) -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama.base_url,
        chat_model=config.ollama.chat_model,
        embed_model=config.ollama.embed_model,
        timeout_sec=config.ollama.timeout_sec,
        num_ctx=config.ollama.num_ctx,
        num_gpu=config.ollama.num_gpu,
        num_batch=config.ollama.num_batch,
        keep_alive=config.ollama.keep_alive,
        manage_vram=config.ollama.manage_vram,
        dry_run=dry_run,
        retry_attempts=config.ollama.retry_attempts,
        retry_backoff_sec=config.ollama.retry_backoff_sec,
        fallback_num_ctx=config.ollama.fallback_num_ctx,
        fallback_num_gpu=config.ollama.fallback_num_gpu,
        fallback_num_batch=config.ollama.fallback_num_batch,
        fallback_max_tokens=config.ollama.fallback_max_tokens,
        fallback_keep_alive=config.ollama.fallback_keep_alive,
    )
