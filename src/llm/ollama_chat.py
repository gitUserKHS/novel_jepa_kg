"""Ollama 채팅 클라이언트 — 기본 백엔드 (Gemma4 26B-A4B, Q4_K_M).

LocalLLMClient(레거시 Qwen 모델 서버)와 같은 표면 — chat / chat_messages / stream_messages / health / status /
count_tokens / dry_run — 을 유지해 생성기·워커·웹이 백엔드를 가리지 않는다. 임베딩은 제공하지 않는다.

Ollama 특성 (2026-09-03 실측, ollama 0.33.2, RTX 4060 8GB, 모델 16.9 GiB 중 VRAM 4.6 GiB, 디코드 ~28 tok/s):
- Gemma4 는 기본으로 생각(thinking)을 켠다. think=false 를 보내지 않으면 답이 전부 thinking 필드로 가고
  content 는 빈다 — 모든 요청에 think=false 를 보낸다.
- num_ctx 는 요청마다 명시한다. 기본 4096 이면 장 프롬프트(~10K 토큰)의 앞부분 — 작품 설정과 인물 — 이
  소리 없이 잘린다. 값을 바꾸면 모델을 다시 올리므로 서비스 안에서는 한 값으로 고정한다.
- 같은 접두사로 시작하는 연속 요청은 KV 캐시를 재사용한다 (8.7K 토큰 프리필 16.4s → 0.4s). 그래서
  json_mode 는 시스템 프롬프트를 건드리지 않고 format="json" + 마지막 사용자 메시지 끝의 힌트로 만든다.
- 한국어 토큰 필터·DRY 샘플러·LoRA 어댑터 같은 로컬 서버 전용 옵션은 받아서 무시한다.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import requests

from src.llm.common import JSON_MODE_HINT, LocalLLMError
from src.llm.dry_run import dry_stream

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# 실측: 한국어 산문 1.5~1.6 자/토큰 (Gemma4 토크나이저). 토큰 수 추정과 문맥 예산 계산에 쓴다.
KOREAN_CHARS_PER_TOKEN = 1.5
# 로컬 모델 서버에만 있는 옵션. Ollama 에는 대응물이 없으니 조용히 버린다.
IGNORED_OPTIONS = frozenset({
    "adapter", "adapter_scale", "use_default_adapter", "korean_filter", "no_repeat_ngram_size",
    "dry_multiplier", "dry_base", "dry_allowed_length",
})


def _normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return normalized[: -len(":latest")] if normalized.endswith(":latest") else normalized


def model_installed(requested: str, installed: list[str]) -> bool:
    wanted = _normalize_model_name(requested)
    return wanted in {_normalize_model_name(name) for name in installed}


class OllamaChatClient:
    backend = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = "",
        *,
        timeout_sec: int = 1800,
        num_ctx: int = 16384,
        keep_alive: str = "30m",
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.05,
        dry_run: bool = False,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        if not model and not dry_run:
            raise ValueError("Ollama 모델 이름이 비어 있어 (configs/default.yaml 의 llm.model).")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = max(10, int(timeout_sec))
        self.num_ctx = max(2048, int(num_ctx))
        self.keep_alive = keep_alive
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.dry_run = bool(dry_run)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        # LocalLLMClient / OllamaClient 호환 속성 (일부 코드가 모델명을 기록한다)
        self.chat_model = model
        self.embed_model = ""
        # 마지막 호출의 토큰 사용량 — 로그·문맥 잘림 감지용
        self.last_usage: dict[str, Any] = {}

    @property
    def display_name(self) -> str:
        return (self.model or "dry-run").rsplit("/", 1)[-1]

    # ---- status -------------------------------------------------------------------------
    def installed_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [str(item.get("name", "")) for item in response.json().get("models", []) if item.get("name")]

    def health(self) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "ok", "backend": self.backend, "model": self.display_name, "dry_run": True}
        try:
            installed = self.installed_models()
        except (requests.RequestException, ValueError) as exc:
            raise LocalLLMError(f"Ollama({self.base_url})에 연결할 수 없어: {exc}") from exc
        if not model_installed(self.model, installed):
            return {"status": "missing", "backend": self.backend, "model": self.display_name, "installed": installed}
        payload: dict[str, Any] = {"status": "ok", "backend": self.backend, "model": self.display_name,
                                   "base_url": self.base_url, "loaded": False}
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            response.raise_for_status()
            for item in response.json().get("models", []):
                if _normalize_model_name(str(item.get("name", ""))) == _normalize_model_name(self.model):
                    payload["loaded"] = True
                    payload["vram_gib"] = round(float(item.get("size_vram", 0)) / 2**30, 1)
                    payload["context_length"] = item.get("context_length")
        except (requests.RequestException, ValueError):
            pass
        return payload

    def is_ready(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except LocalLLMError:
            return False

    def status(self) -> dict[str, Any]:
        """UI 용 요약: ready 여부와 사유."""
        try:
            payload = self.health()
        except LocalLLMError as exc:
            return {"ready": False, "reason": str(exc), "health": None}
        if payload.get("status") != "ok":
            return {
                "ready": False,
                "reason": f"Ollama 에 모델 '{self.model}' 이 없어. 터미널에서 `ollama pull {self.model}` 을 실행해줘.",
                "health": payload,
            }
        return {"ready": True, "reason": "", "health": payload}

    # ---- generation ---------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        stream_callback: Callable[[str], None] | None = None,
        **options: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                                  stream_callback=stream_callback, **options)

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        stream_callback: Callable[[str], None] | None = None,
        **options: Any,
    ) -> str:
        pieces: list[str] = []
        for piece in self.stream_messages(messages, temperature=temperature, max_tokens=max_tokens,
                                          json_mode=json_mode, **options):
            pieces.append(piece)
            if stream_callback is not None:
                stream_callback(piece)
        text = "".join(pieces)
        if not text.strip():
            raise LocalLLMError(
                f"모델({self.display_name})이 빈 응답을 돌려줬어. done_reason={self.last_usage.get('done_reason')!r}. "
                "max_tokens 를 늘리거나 프롬프트를 줄여봐."
            )
        return text

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_p: float = 0.0,
        **ignored: Any,
    ) -> Iterator[str]:
        """조각 단위 제너레이터. Streamlit `st.write_stream` 에 그대로 넘길 수 있다."""
        unknown = set(ignored) - IGNORED_OPTIONS
        if unknown:
            logger.debug("Ollama client ignoring unsupported options: %s", sorted(unknown))
        payload_messages = self._json_mode_messages(messages) if json_mode else [dict(m) for m in messages]
        if self.dry_run:
            prompt = payload_messages[-1]["content"] if payload_messages else ""
            yield from dry_stream(prompt, json_mode)
            return
        body = self._build_body(payload_messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                                seed=seed, stop=stop, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                                min_p=min_p)
        attempts = 1 + self.retry_attempts
        last_error: Exception | None = None
        for attempt in range(attempts):
            delivered = False
            try:
                with requests.post(f"{self.base_url}/api/chat", json=body, stream=True,
                                   timeout=(10, self.timeout_sec)) as response:
                    if response.status_code != 200:
                        raise LocalLLMError(f"Ollama 오류 HTTP {response.status_code}: {response.text[:300]}")
                    for piece, final in _iter_ndjson(response):
                        if final is not None:
                            self._record_usage(final, max_tokens)
                            break
                        if piece:
                            delivered = True
                            yield piece
                return
            except (requests.RequestException, LocalLLMError) as exc:
                last_error = exc
                if delivered or attempt >= attempts - 1:
                    break
                time.sleep(self.retry_backoff_sec * (attempt + 1))
        raise LocalLLMError(f"Ollama 호출 실패 ({self.base_url}, {self.display_name}): {last_error}")

    def count_tokens(self, messages: list[dict[str, str]] | None = None, text: str | None = None) -> int:
        """토큰 수 추정. Ollama 에는 공개 토크나이즈 API 가 없어 실측 비율(한국어 1.5자/토큰)로 계산한다."""
        source = text if text is not None else "\n".join(str(m.get("content", "")) for m in (messages or []))
        return max(1, round(len(source) / KOREAN_CHARS_PER_TOKEN))

    def embed(self, texts: list[str], **_kwargs: Any):  # noqa: ANN201 - OllamaClient 호환 시그니처
        raise LocalLLMError("소설 백엔드는 임베딩을 제공하지 않아 (JEPA/RAG 인덱스 비활성).")

    # ---- internals ----------------------------------------------------------------------
    @staticmethod
    def _json_mode_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """JSON 힌트를 마지막 사용자 메시지 끝에 붙인다. 시스템 프롬프트는 접두사 캐시를 위해 그대로 둔다."""
        payload = [dict(m) for m in messages]
        for message in reversed(payload):
            if message.get("role") == "user":
                message["content"] = message["content"].rstrip() + "\n\n" + JSON_MODE_HINT
                break
        else:
            payload.append({"role": "user", "content": JSON_MODE_HINT})
        return payload

    def _build_body(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        seed: int | None,
        stop: list[str] | None,
        top_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        min_p: float,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(self.top_p if top_p is None else top_p),
            "top_k": int(self.top_k if top_k is None else top_k),
            "num_predict": int(max_tokens),
            "num_ctx": int(self.num_ctx),
            "repeat_penalty": float(self.repetition_penalty if repetition_penalty is None else repetition_penalty),
        }
        if seed is not None:
            options["seed"] = int(seed)
        if stop:
            options["stop"] = [str(item) for item in stop]
        if min_p and float(min_p) > 0:
            options["min_p"] = float(min_p)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,  # Gemma4: 이게 없으면 답이 thinking 으로만 간다 (09-03 실측)
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        return body

    def _record_usage(self, final: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        prompt_tokens = int(final.get("prompt_eval_count") or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(final.get("eval_count") or 0),
            "prompt_seconds": round(float(final.get("prompt_eval_duration") or 0) / 1e9, 2),
            "output_seconds": round(float(final.get("eval_duration") or 0) / 1e9, 2),
            "load_seconds": round(float(final.get("load_duration") or 0) / 1e9, 2),
            "done_reason": final.get("done_reason"),
            # Ollama 는 num_ctx 를 넘는 프롬프트를 앞에서부터 소리 없이 자른다. 잘렸다면 평가된 프롬프트 토큰이
            # 창을 거의 채운다 — 그 조합을 보고 경고한다.
            "truncated": prompt_tokens > 0 and prompt_tokens + int(max_tokens) >= self.num_ctx - 16,
        }
        if usage["truncated"]:
            logger.warning(
                "Ollama prompt likely truncated: %d prompt tokens + %d output >= num_ctx %d. "
                "작품 설정·인물이 잘렸을 수 있다 — llm.num_ctx 를 늘리거나 문맥 예산을 줄여라.",
                prompt_tokens, max_tokens, self.num_ctx,
            )
        self.last_usage = usage
        return usage


def _iter_ndjson(response: requests.Response) -> Iterator[tuple[str, dict[str, Any] | None]]:
    """(본문 조각, 최종 페이로드) 쌍을 낸다. 최종 페이로드는 done=true 줄에서만 non-None. thinking 조각은 버린다."""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("error"):
            raise LocalLLMError(f"Ollama 오류: {payload['error']}")
        message = payload.get("message") or {}
        content = str(message.get("content") or "")
        if content:
            yield content, None
        if payload.get("done"):
            yield "", payload
            return
