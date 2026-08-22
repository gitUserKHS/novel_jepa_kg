from __future__ import annotations

from src.llm.local_client import LocalLLMClient
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig


def make_llm_client(config: AppConfig, *, dry_run: bool = False) -> LocalLLMClient | OllamaClient:
    """config.llm.backend 에 따라 로컬 Qwen 서버 또는 (레거시) Ollama 클라이언트."""
    if config.llm.backend == "ollama":
        return make_ollama_client(config, dry_run=dry_run)
    return LocalLLMClient(
        config.llm.base_url,
        timeout_sec=config.llm.timeout_sec,
        default_adapter=config.llm.chat_adapter or None,
        adapter_scale=config.llm.adapter_scale,
        korean_filter=config.llm.korean_filter,
        top_p=config.llm.novel_top_p,
        top_k=config.llm.novel_top_k,
        repetition_penalty=config.llm.novel_repetition_penalty,
        dry_run=dry_run or config.llm.dry_run,
    )


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
        top_p=config.ollama.top_p,
        repeat_penalty=config.ollama.repeat_penalty,
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
