"""LLM 클라이언트 공통 — 오류 타입과 JSON 모드 힌트. 백엔드(Ollama / 레거시 로컬 서버)가 함께 쓴다."""

from __future__ import annotations


class LocalLLMError(RuntimeError):
    """로컬 모델(Ollama 또는 레거시 모델 서버) 호출 실패. 웹·워커는 이 하나만 잡는다."""


JSON_MODE_HINT = (
    "반드시 JSON 객체 하나만 출력한다. 앞뒤 설명, 사과문, 주석, 마크다운 코드펜스를 쓰지 않는다. "
    "첫 글자는 { 이고 마지막 글자는 } 이다."
)
