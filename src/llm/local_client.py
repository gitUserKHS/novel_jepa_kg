"""로컬 Qwen 모델 서버 클라이언트 (model_server/server.py 의 OpenAI 호환 API).

OllamaClient 와 같은 표면(chat/stream_callback/json_mode/dry_run)을 유지해 기존 생성
코드가 그대로 붙는다. 임베딩은 제공하지 않는다 — JEPA/RAG 인덱스 경로는 이
백엔드에서 비활성화된다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import requests

LOCAL_MODEL_NAME = "qwen3.5-4b-4bit"
JSON_MODE_HINT = (
    "반드시 JSON 객체 하나만 출력한다. 앞뒤 설명, 사과문, 주석, 마크다운 코드펜스를 쓰지 않는다. "
    "첫 글자는 { 이고 마지막 글자는 } 이다."
)


class LocalLLMError(RuntimeError):
    pass


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        timeout_sec: int = 900,
        default_adapter: str | None = None,
        adapter_scale: float = 0.6,
        korean_filter: bool = True,
        top_p: float = 0.9,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        dry_run: bool = False,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(10, int(timeout_sec))
        self.default_adapter = default_adapter or None
        self.adapter_scale = float(adapter_scale)
        self.korean_filter = bool(korean_filter)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.dry_run = bool(dry_run)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        # OllamaClient 호환 속성 (일부 코드가 모델명을 기록한다)
        self.chat_model = LOCAL_MODEL_NAME
        self.embed_model = ""

    # ---- status -------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "ok", "model": LOCAL_MODEL_NAME, "adapters": ["cute"], "dry_run": True}
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise LocalLLMError(f"모델 서버({self.base_url})에 연결할 수 없어: {exc}") from exc

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
            return {"ready": False, "reason": "모델을 불러오는 중이야.", "health": payload}
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
        return self.chat_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream_callback=stream_callback,
            **options,
        )

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
        for piece in self.stream_messages(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, **options
        ):
            pieces.append(piece)
            if stream_callback is not None:
                stream_callback(piece)
        text = "".join(pieces)
        if not text.strip():
            raise LocalLLMError("모델 서버가 빈 응답을 돌려줬어. max_tokens 를 늘리거나 프롬프트를 줄여봐.")
        return text

    def stream_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        adapter: str | None = None,
        adapter_scale: float | None = None,
        korean_filter: bool | str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int = 0,
        seed: int | None = None,
        stop: list[str] | None = None,
        use_default_adapter: bool = False,
        min_p: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
    ) -> Iterator[str]:
        """조각 단위 제너레이터. Streamlit `st.write_stream` 에 그대로 넘길 수 있다."""
        payload_messages = [dict(m) for m in messages]
        if json_mode:
            if payload_messages and payload_messages[0]["role"] == "system":
                payload_messages[0]["content"] = payload_messages[0]["content"].rstrip() + "\n" + JSON_MODE_HINT
            else:
                payload_messages.insert(0, {"role": "system", "content": JSON_MODE_HINT})
        if self.dry_run:
            yield from self._dry_stream(payload_messages, json_mode)
            return
        chosen_adapter = adapter if adapter is not None else (self.default_adapter if use_default_adapter else None)
        body: dict[str, Any] = {
            "messages": payload_messages,
            "temperature": float(temperature),
            "top_p": float(self.top_p if top_p is None else top_p),
            "top_k": int(self.top_k if top_k is None else top_k),
            "max_tokens": int(max_tokens),
            "stream": True,
            "adapter": chosen_adapter or None,
            "adapter_scale": float(self.adapter_scale if adapter_scale is None else adapter_scale),
            "korean_filter": (self.korean_filter if korean_filter is None else korean_filter),
            "repetition_penalty": float(self.repetition_penalty if repetition_penalty is None else repetition_penalty),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "seed": seed,
            "stop": list(stop or []),
            "min_p": float(min_p),
            "dry_multiplier": float(dry_multiplier),
            "dry_base": float(dry_base),
            "dry_allowed_length": int(dry_allowed_length),
        }
        attempts = 1 + self.retry_attempts
        last_error: Exception | None = None
        for attempt in range(attempts):
            delivered = False
            try:
                with requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    stream=True,
                    timeout=(10, self.timeout_sec),
                ) as response:
                    if response.status_code != 200:
                        raise LocalLLMError(
                            f"모델 서버 오류 HTTP {response.status_code}: {response.text[:300]}"
                        )
                    final_text: str | None = None
                    for piece, final in _iter_sse(response):
                        if final is not None:
                            final_text = final
                            break
                        if piece:
                            delivered = True
                            yield piece
                    if final_text is not None:
                        # 정지 문자열로 잘린 경우 서버가 최종 본문을 보낸다 — 이미 보낸
                        # 조각보다 짧을 수 있으므로 호출자는 stop 을 쓸 때 결과를 재조립한다.
                        pass
                return
            except (requests.RequestException, LocalLLMError) as exc:
                last_error = exc
                if delivered or attempt >= attempts - 1:
                    break
                time.sleep(self.retry_backoff_sec * (attempt + 1))
        raise LocalLLMError(f"모델 서버 호출 실패 ({self.base_url}): {last_error}")

    def count_tokens(self, messages: list[dict[str, str]] | None = None, text: str | None = None) -> int:
        if self.dry_run:
            source = text if text is not None else json.dumps(messages or [], ensure_ascii=False)
            return max(1, len(source) // 2)
        payload: dict[str, Any] = {"messages": messages} if messages is not None else {"text": text or ""}
        try:
            response = requests.post(f"{self.base_url}/v1/count_tokens", json=payload, timeout=30)
            response.raise_for_status()
            return int(response.json()["tokens"])
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise LocalLLMError(f"토큰 계산 실패: {exc}") from exc

    def embed(self, texts: list[str], **_kwargs: Any):  # noqa: ANN201 - OllamaClient 호환 시그니처
        raise LocalLLMError("로컬 Qwen 백엔드는 임베딩을 제공하지 않아 (JEPA/RAG 인덱스 비활성).")

    # ---- dry run --------------------------------------------------------------------------
    def _dry_stream(self, messages: list[dict[str, str]], json_mode: bool) -> Iterator[str]:
        prompt = messages[-1]["content"] if messages else ""
        if json_mode or "JSON" in prompt:
            text = json.dumps(
                {
                    "reply": "설정을 정리했어. 이대로 시작할까?",
                    "ready": True,
                    "card": {
                        "title": "유리등의 속삭임",
                        "genre": "SF 미스터리",
                        "premise": "비가 멈춘 도시에서 사라진 기억을 추적한다",
                        "world": "기억이 거래되는 근미래 서울",
                        "protagonist": "서윤: 기록 복원가",
                        "characters": "민재: 핵심 단서를 숨긴 연구원",
                        "target_chars": 30000,
                    },
                    "premise": "주인공의 선택이 도시의 균형을 바꾼다",
                    "ending_intent": "대가를 치르고 진실을 택한다",
                    "beats": [
                        {"beat_id": 1, "phase": "1막", "purpose": "균형을 깨는 사건", "required_change": "주인공이 문제를 외면할 수 없게 된다"},
                        {"beat_id": 2, "phase": "2막", "purpose": "선택과 대가", "required_change": "동맹의 조건이 달라진다"},
                        {"beat_id": 3, "phase": "3막", "purpose": "수렴과 결말", "required_change": "중심 질문에 답한다"},
                    ],
                    "section_index": 1,
                    "summary": "서윤이 첫 단서를 얻는다.",
                    "characters_list": ["서윤", "민재"],
                    "facts": ["기록 장치가 동생의 목소리를 재생했다"],
                    "open_clues": ["심층 구역 좌표"],
                    "resolved_clues": [],
                    "locations": ["폐쇄 연구동"],
                    "state_changes": ["서윤이 좌표를 확보했다"],
                    "keywords": ["기록", "좌표", "동생"],
                },
                ensure_ascii=False,
            )
            yield text
            return
        text = (
            "### 젖은 골목의 신호\n\n"
            "서윤은 차가운 형광등 아래에서 숨을 골랐다. 기록 장치가 토해낸 잔향은 동생의 목소리였지만, "
            "그 안에는 도망치는 사람의 공포보다 무언가를 선택한 사람의 단단함이 남아 있었다.\n\n"
            "민재는 심층 구역 좌표를 보는 순간 얼굴빛을 잃었다. 서윤은 그 침묵이 대답이라는 걸 알았다. "
            "이제 그녀가 찾아야 할 것은 동생의 행방만이 아니었다. 왜 모두가 그 선택을 숨기려 했는지, "
            "그리고 자신이 잃어버린 첫 번째 기억이 무엇인지 확인해야 했다.\n\n"
            "비는 그치지 않았다. 서윤은 젖은 골목 끝에서 다시 한 번 장치를 켰다."
        )
        for start in range(0, len(text), 24):
            yield text[start : start + 24]


def _iter_sse(response: requests.Response) -> Iterator[tuple[str, str | None]]:
    """(조각, 최종본문) 쌍을 낸다. 최종본문은 정지 문자열로 잘렸을 때만 non-None."""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if "error" in payload:
            raise LocalLLMError(f"모델 서버 오류: {payload['error']}")
        choices = payload.get("choices") or []
        delta = ""
        if choices:
            delta = str(choices[0].get("delta", {}).get("content") or "")
            if choices[0].get("finish_reason") and payload.get("text") is not None:
                yield "", str(payload["text"])
                continue
        if delta:
            yield delta, None
