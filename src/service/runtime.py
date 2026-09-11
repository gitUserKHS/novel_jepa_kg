"""LLM 클라이언트 팩토리. 소설 생성·기획·퇴고의 모든 모델 호출은 여기서 만든 클라이언트를 쓴다 (AGENTS.md).

- `ollama` (기본): Ollama 의 Gemma4 26B-A4B — `src/llm/ollama_chat.OllamaChatClient`
- `local`: 레거시 Qwen3.5-4B 모델 서버(`model_server/server.py`) — `src/llm/local_client.LocalLLMClient`
- `make_ollama_client`: 레거시 연구 UI(`app.py`) 전용 임베딩 포함 클라이언트. 소설 경로에서는 쓰지 않는다.
"""

from __future__ import annotations

from src.llm.local_client import LocalLLMClient
from src.llm.ollama_chat import OllamaChatClient
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig

NovelClient = OllamaChatClient | LocalLLMClient


def make_llm_client(config: AppConfig, *, dry_run: bool = False) -> NovelClient:
    backend = str(config.llm.backend or "ollama").strip().lower()
    dry = bool(dry_run or config.llm.dry_run)
    if backend == "ollama":
        return OllamaChatClient(
            config.llm.ollama_base_url,
            config.llm.model,
            timeout_sec=config.llm.timeout_sec,
            num_ctx=config.llm.num_ctx,
            keep_alive=config.llm.keep_alive,
            top_p=config.llm.novel_top_p,
            top_k=config.llm.novel_top_k,
            repetition_penalty=config.llm.novel_repetition_penalty,
            dry_run=dry,
        )
    if backend == "local":
        return LocalLLMClient(
            config.llm.base_url,
            timeout_sec=config.llm.timeout_sec,
            korean_filter=config.llm.korean_filter,
            top_p=config.llm.novel_top_p,
            top_k=config.llm.novel_top_k,
            repetition_penalty=config.llm.novel_repetition_penalty,
            dry_run=dry,
        )
    raise ValueError(
        f"알 수 없는 llm.backend: {backend!r}. 'ollama'(기본, Gemma4 26B-A4B) 또는 'local'(레거시 Qwen 모델 서버) 중 하나여야 해."
    )


def make_ollama_client(config: AppConfig, *, dry_run: bool = False) -> OllamaClient:
    """레거시 연구 UI 용 (임베딩·JEPA). configs 의 `ollama:` 절을 쓴다."""
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
