from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm.ollama_client import OllamaClient
from src.utils.paths import ensure_parent


class OutlineBeat(BaseModel):
    beat_id: int = Field(ge=1)
    phase: str
    purpose: str
    required_change: str
    setup_or_payoff: str = ""
    forbidden_repeat: str = ""


class StoryOutline(BaseModel):
    premise: str
    ending_intent: str
    beats: list[OutlineBeat]


def create_story_outline(
    client: OllamaClient,
    *,
    world: str,
    characters: str,
    premise: str,
    target_chars: int,
    beat_count: int = 12,
) -> StoryOutline:
    safe_count = max(6, min(20, int(beat_count)))
    prompt = f"""
다음 한국어 장편 소설을 위한 계층형 이야기 지도를 설계하세요.

[세계관]
{world}

[인물표]
{characters}

[핵심 소재와 시작점]
{premise}

[목표]
- 전체 분량: 약 {int(target_chars):,}자
- 정확히 {safe_count}개의 순차 beat
- 1막 설정과 압박, 2막 선택과 대가, 3막 수렴과 결말 의도를 모두 포함
- 각 beat는 사건을 하나만 전진시키고 이전 beat의 결과를 이어받음
- 인물표에 없는 새 고유명사 인물은 만들지 않음
- 단서의 setup과 payoff를 분리하고, 같은 반전이나 경고를 새 사실처럼 반복하지 않음

JSON 객체 하나만 출력하세요. 마크다운 코드 블록은 쓰지 마세요.
{{
  "premise": "작품의 변하지 않는 중심 질문",
  "ending_intent": "결말이 수렴할 정서와 해결 방향",
  "beats": [
    {{
      "beat_id": 1,
      "phase": "1막",
      "purpose": "이번 beat의 서사 기능",
      "required_change": "끝날 때 반드시 달라져야 하는 상태",
      "setup_or_payoff": "설치하거나 회수할 단서",
      "forbidden_repeat": "이후 새 사실처럼 반복하면 안 되는 내용"
    }}
  ]
}}
""".strip()
    try:
        raw = client.chat(
            prompt,
            system="당신은 장편 소설의 인과관계와 복선을 설계하는 한국어 스토리 에디터입니다.",
            temperature=0.25,
            max_tokens=1100,
        )
        outline = StoryOutline.model_validate(_json_object(raw))
        if len(outline.beats) < 4:
            raise ValueError("outline contains too few beats")
        outline.beats = [
            beat.model_copy(update={"beat_id": index})
            for index, beat in enumerate(outline.beats[:safe_count], start=1)
        ]
        return outline
    except Exception:
        return fallback_story_outline(premise, safe_count)


def fallback_story_outline(premise: str, beat_count: int = 12) -> StoryOutline:
    templates = [
        ("1막", "일상의 균형을 깨는 구체적 사건", "주인공이 문제를 외면할 수 없게 된다"),
        ("1막", "목표와 위험의 범위를 확인", "주인공이 첫 선택을 내린다"),
        ("1막", "첫 행동의 예상 밖 결과", "기존 계획이 충분하지 않음이 드러난다"),
        ("2막", "새 단서의 검증", "추측 하나가 사실 또는 오해로 판명된다"),
        ("2막", "관계의 이해관계 충돌", "동맹이나 신뢰의 조건이 달라진다"),
        ("2막", "선택의 대가를 현실화", "인물·장소·자원 중 하나의 상태가 바뀐다"),
        ("2막", "중앙 전환점에서 목표 재정의", "주인공이 진짜 문제를 새로 이해한다"),
        ("2막", "적대 세력이나 환경의 반격", "안전한 선택지가 하나 사라진다"),
        ("2막", "초반 복선의 부분 회수", "새 행동을 가능하게 하는 구체적 정보가 생긴다"),
        ("3막", "최종 선택 직전의 손실과 결단", "주인공이 감수할 대가를 명확히 선택한다"),
        ("3막", "핵심 갈등의 행동 중심 해결", "중심 질문에 되돌릴 수 없는 답을 만든다"),
        ("3막", "결과와 정서적 여운", "변화한 세계와 관계를 구체적 장면으로 보여준다"),
    ]
    beats: list[OutlineBeat] = []
    for index in range(max(6, int(beat_count))):
        phase, purpose, change = templates[min(index, len(templates) - 1)]
        beats.append(
            OutlineBeat(
                beat_id=index + 1,
                phase=phase,
                purpose=purpose,
                required_change=change,
                setup_or_payoff="이전 단서의 인과를 이어서 설치하거나 회수한다",
                forbidden_repeat="이미 확정된 사실을 새 반전처럼 다시 설명하지 않는다",
            )
        )
    return StoryOutline(
        premise=premise.strip() or "주인공의 선택이 세계와 관계를 어떻게 바꾸는가",
        ending_intent="핵심 갈등의 결과를 행동으로 확정하고 변화한 관계의 여운을 남긴다",
        beats=beats,
    )


def outline_context(
    outline: StoryOutline,
    *,
    section_index: int,
    planned_sections: int,
) -> tuple[str, int]:
    if not outline.beats:
        return "(story outline unavailable)", 0
    progress = max(0.0, min(0.999999, (max(1, section_index) - 1) / max(1, planned_sections)))
    beat_index = min(len(outline.beats) - 1, int(progress * len(outline.beats)))
    current = outline.beats[beat_index]
    previous = outline.beats[beat_index - 1] if beat_index > 0 else None
    upcoming = outline.beats[beat_index + 1] if beat_index + 1 < len(outline.beats) else None
    lines = [
        "[Global story spine]",
        f"- central premise: {outline.premise}",
        f"- ending intent: {outline.ending_intent}",
        f"- active beat {current.beat_id}/{len(outline.beats)} ({current.phase}): {current.purpose}",
        f"- required state change: {current.required_change}",
    ]
    if current.setup_or_payoff:
        lines.append(f"- setup/payoff: {current.setup_or_payoff}")
    if current.forbidden_repeat:
        lines.append(f"- do not repeat: {current.forbidden_repeat}")
    if previous:
        lines.append(f"- previous beat consequence to preserve: {previous.required_change}")
    if upcoming:
        lines.append(f"- prepare, but do not complete yet: {upcoming.purpose}")
    return "\n".join(lines), beat_index


def load_story_outline(path: Path) -> StoryOutline | None:
    if not path.exists():
        return None
    try:
        return StoryOutline.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_story_outline(path: Path, outline: StoryOutline) -> str:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(outline.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


def _json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("outline response did not contain a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("outline JSON must be an object")
    return payload
