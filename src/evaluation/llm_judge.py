from __future__ import annotations

import logging
from typing import Any, Protocol

from src.data.validate_jsonl import parse_json_object

logger = logging.getLogger(__name__)

# Rubric dimensions scored 1-10 by the judge model. `hallucination_control` is
# the service-defining axis: invented detail should enrich the story without
# betraying the established world, characters, or prior scenes.
JUDGE_SCORE_KEYS = (
    "plausibility",
    "creativity",
    "hallucination_control",
    "consistency",
    "immersion",
)

JUDGE_SCORE_LABELS = {
    "plausibility": "개연성",
    "creativity": "창의성",
    "hallucination_control": "할루시네이션 통제",
    "consistency": "설정 일관성",
    "immersion": "몰입도",
}

_MAX_FINDING_ITEMS = 5
_MAX_FINDING_CHARS = 200
_MAX_VERDICT_CHARS = 300

_JUDGE_SYSTEM = (
    "당신은 한국어 장편 소설 심사위원이다. 이 소설은 '통제된 할루시네이션' 방식으로 생성되었다. "
    "모델이 지어낸 새로운 단서, 상징, 감각 묘사는 의도된 창작 기법이므로 새로움 자체를 감점하지 마라. "
    "대신 지어낸 세부가 기존 설정과 이전 장면을 배신하는지, 아니면 이야기를 풍부하게 만드는지를 구분하라. "
    "반드시 JSON 객체 하나만 출력하라."
)


class JudgeClient(Protocol):
    dry_run: bool

    def chat(
        self,
        prompt: str,
        system: str | None = ...,
        temperature: float = ...,
        max_tokens: int = ...,
        json_mode: bool = ...,
    ) -> str: ...


def judge_excerpt(text: str, max_chars: int = 4000) -> str:
    """Bound long prose to a head/tail excerpt that fits the judge context."""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return "\n\n[...중략...]\n\n".join([cleaned[:half], cleaned[-half:]])


def _judge_prompt(
    previous_scene: str,
    excerpt: str,
    world: str,
    characters: str,
) -> str:
    lines = [
        "다음은 통제된 할루시네이션 방식으로 생성된 한국어 소설 본문이다.",
        "",
        f"[이전 장면]\n{previous_scene.strip() or '(제공되지 않음)'}",
    ]
    if world.strip():
        lines.append(f"\n[세계관 설정]\n{world.strip()}")
    if characters.strip():
        lines.append(f"\n[등장인물 설정]\n{characters.strip()}")
    lines.extend(
        [
            f"\n[생성된 본문 (발췌)]\n{excerpt}",
            "",
            "아래 다섯 항목을 각각 1~10 정수로 채점하라.",
            "- plausibility: 이전 장면에서 자연스럽게 이어지는가 (개연성)",
            "- creativity: 지어낸 세부가 신선하고 생생한가 (창의성)",
            "- hallucination_control: 지어낸 세부가 설정을 배신하지 않고 이야기를 풍부하게 하는가",
            "- consistency: 세계관과 인물 설정을 지키는가",
            "- immersion: 문장과 장면이 독자를 끌어들이는가",
            "",
            "또한 다음을 함께 보고하라.",
            "- useful_hallucinations: 이야기를 풍부하게 만든 지어낸 세부 목록 (최대 5개, 각 한 문장)",
            "- harmful_hallucinations: 설정이나 이전 장면과 모순되는 지어낸 세부 목록 (최대 5개, 각 한 문장)",
            "- verdict: 한두 문장의 총평",
            "",
            "다음 형식의 JSON 객체 하나만 출력하라:",
            '{"plausibility": 7, "creativity": 8, "hallucination_control": 7, '
            '"consistency": 8, "immersion": 7, "useful_hallucinations": ["..."], '
            '"harmful_hallucinations": [], "verdict": "..."}',
        ]
    )
    return "\n".join(lines)


def _clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return max(1.0, min(10.0, score))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text[:_MAX_FINDING_CHARS])
        if len(items) >= _MAX_FINDING_ITEMS:
            break
    return items


def _unavailable(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "backend": "ollama",
        "error": error,
        "scores": {},
        "overall": 0.0,
        "useful_hallucinations": [],
        "harmful_hallucinations": [],
        "verdict": "",
    }


def _dry_run_result() -> dict[str, Any]:
    scores = {key: 7.0 for key in JUDGE_SCORE_KEYS}
    return {
        "available": True,
        "backend": "dry-run",
        "scores": scores,
        "overall": 0.7,
        "useful_hallucinations": ["드라이런 모드: 고정 심사 결과입니다."],
        "harmful_hallucinations": [],
        "verdict": "드라이런 모드에서는 실제 심사 없이 고정 점수를 반환합니다.",
    }


def run_llm_judge(
    client: JudgeClient,
    previous_scene: str,
    generated: str,
    *,
    world: str = "",
    characters: str = "",
    temperature: float = 0.2,
    max_tokens: int = 700,
    excerpt_chars: int = 4000,
) -> dict[str, Any]:
    """Score generated prose with the local chat model as an optional judge.

    Judge failure must never lose the rest of the evaluation report, so every
    backend or parsing problem downgrades to `available: False` with an error
    instead of raising.
    """
    if not generated.strip():
        return _unavailable("empty output")
    if getattr(client, "dry_run", False):
        return _dry_run_result()
    prompt = _judge_prompt(previous_scene, judge_excerpt(generated, excerpt_chars), world, characters)
    try:
        raw = client.chat(
            prompt,
            system=_JUDGE_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        payload = parse_json_object(raw)
    except Exception as exc:  # noqa: BLE001 - the judge is optional by design.
        logger.warning("LLM judge unavailable: %s", exc)
        return _unavailable(str(exc))

    scores = {}
    for key in JUDGE_SCORE_KEYS:
        score = _clamp_score(payload.get(key))
        if score is not None:
            scores[key] = score
    if not scores:
        logger.warning("LLM judge returned no usable scores.")
        return _unavailable("judge returned no usable scores")

    return {
        "available": True,
        "backend": "ollama",
        "scores": scores,
        "overall": round(sum(scores.values()) / (10.0 * len(scores)), 4),
        "useful_hallucinations": _string_list(payload.get("useful_hallucinations")),
        "harmful_hallucinations": _string_list(payload.get("harmful_hallucinations")),
        "verdict": str(payload.get("verdict", "")).strip()[:_MAX_VERDICT_CHARS],
    }
