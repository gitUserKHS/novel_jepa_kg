"""장 단위 개연성 검토 — 연속성 편집자 역할의 모델 호출.

새 장이 (1) 확정된 설정·인물, (2) 상태 원장과 앞선 사건, (3) 직전 장면의 끝, (4) 이번 장의 설계와 어긋나는지
모델에게 묻고 JSON 으로 받는다. 모순이 하나라도 있거나 점수가 기준 미만이면 생성기가 그 목록을 들고 장을
고쳐 쓴 뒤 다시 검토한다 (longform.generate_longform). 검토 자체가 실패하면(연결 오류·JSON 아님)
available=False 로 돌려 본문을 막지 않는다 — 검토는 문지기지 병목이 아니다.

프롬프트는 생성기의 공통 접두사(작품 설정 → 인물 → 이야기 지도 → 줄거리·상태 → 지나간 사건 → 직전 장면 끝)
뒤에 검토 대상과 역할 지시를 붙인다. Ollama 는 같은 접두사의 KV 캐시를 재사용하므로 검토 호출의 프리필 비용은
거의 0 이다 (09-03 실측 16.4s → 0.4s).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.llm.jsonish import json_object

logger = logging.getLogger(__name__)

REVIEW_ROLE_MARKER = "[역할: 연속성 편집자]"
DIALOGUE_ROLE_MARKER = "[역할: 대사 화자 점검]"
MAX_ITEMS = 4
MAX_ITEM_CHARS = 120
QUOTE_CHARS = ('"', "“", "”", "「", "『")


class PlausibilityReport(BaseModel):
    score: int = Field(default=10, ge=1, le=10)
    contradictions: list[str] = Field(default_factory=list)  # 설정·앞선 사건·상태 원장과의 모순
    unmotivated: list[str] = Field(default_factory=list)  # 원인 없는 사건, 동기 없는 행동, 편의적 우연
    continuity_ok: bool = True  # 직전 장면 끝에서 시간·장소·인물 상태가 이어지는가
    beat_progress: bool = True  # 이번 장의 설계·요청이 실제로 일어났는가
    verdict: str = ""
    available: bool = True  # False = 검토를 못 했다 (점수·목록은 의미 없음)

    @classmethod
    def unavailable(cls, reason: str) -> "PlausibilityReport":
        return cls(score=10, verdict=reason[:200], available=False)

    def issues(self, min_score: int) -> list[str]:
        """고쳐 써야 하는 문제 (한국어 한 줄씩). 비어 있으면 통과."""
        if not self.available:
            return []
        found = [f"개연성: 확정 사실과 모순 — {item}" for item in self.contradictions[:MAX_ITEMS]]
        if self.score < int(min_score):
            found.extend(f"개연성: 원인 없는 사건 — {item}" for item in self.unmotivated[:MAX_ITEMS])
            if not self.continuity_ok:
                found.append("개연성: 직전 장면의 끝에서 자연스럽게 이어지지 않음")
            if not self.beat_progress:
                found.append("개연성: 이번 장의 설계(사용자 요청)가 실제로 일어나지 않음")
            if not found:
                found.append(f"개연성 점수 {self.score}/10 (기준 {int(min_score)}) — {self.verdict or '설명 없음'}")
        return found

    def passes(self, min_score: int) -> bool:
        return not self.issues(min_score)

    def notes(self) -> list[str]:
        """통과했더라도 기록으로 남길 연한 관찰."""
        if not self.available:
            return []
        out = [f"주의: {item}" for item in self.unmotivated[:2]]
        if not self.beat_progress:
            out.append("주의: 이번 장의 설계가 온전히 실현되지 않음")
        return out

    def summary(self) -> str:
        if not self.available:
            return f"개연성 검토 불가 ({self.verdict})"
        detail = self.verdict or ("모순 없음" if not self.contradictions else "; ".join(self.contradictions[:2]))
        return f"개연성 {self.score}/10 — {detail}"


def review_prompt(prefix: str, *, section: str, section_index: int, plan_text: str, instruction: str) -> str:
    plan_block = plan_text.strip() or f"(설계 없음 — 사용자 요청: {instruction.strip() or '이야기를 한 걸음 옮긴다'})"
    return "\n".join([
        prefix.rstrip(),
        f"[이번 장({section_index}장) 본문 — 검토 대상]",
        section.strip()[:6000],
        "[이번 장의 설계 — 이 장이 하기로 한 것]",
        plan_block,
        f"{REVIEW_ROLE_MARKER} 위 장이 앞선 이야기와 맞물리는지 검토한다. 당신은 이 장을 쓴 사람이 아니라 트집을 잡는 편이 "
        "일인 편집자다. 새 사건·새 묘사 자체는 문제가 아니지만, 문장 하나하나를 의심하며 읽는다. 다음 네 가지를 본다.",
        "1. 모순: [작품 설정]·[인물]·[지금까지의 줄거리와 현재 상태]에 확정된 사실(장소·소유·관계·목표·생사·시간·정체)과 어긋나는 "
        "서술. 예: 동물이나 물건이 사람 말을 함, 죽었거나 자리에 없는 인물이 등장·발언함, 인물이 알 수 없는 정보를 이미 알고 있음, "
        "장소·시간이 설명 없이 바뀜, 이미 해결된 단서를 새 사실처럼 다시 꺼냄, 인물표에 없는 인물이 이름을 갖고 등장함. "
        "누가 한 말인지 문맥상 잘못 읽히는 대사(말할 수 없는 존재 바로 뒤에 붙은 대사)도 모순으로 적는다.",
        "2. 원인 없는 사건: 앞선 사건이나 이 장 안의 원인 없이 갑자기 일어나는 사건, 동기 없는 행동·태도 급변, 편의적인 우연, "
        "근거 없이 단정하는 추론.",
        "3. 연속성: [직전 장면의 끝부분]에서 시간·장소·인물 상태·소지품이 그대로 이어지는가.",
        "4. 전진: 위 설계(또는 사용자 요청)가 실제로 일어나 이야기가 한 걸음 나아갔는가.",
        "절차: 먼저 의심스러운 대목을 최소 두 개 골라 원문 구절을 떠올리고, 각각이 위 기준에 걸리는지 판정한 뒤에 점수를 매긴다. "
        "걸리는 것이 하나도 없을 때만 목록이 비어 있다. (대사의 화자 배정은 따로 점검하므로 여기서는 반복하지 않는다.)",
        "점수: 10 = 의심스러운 대목을 찾았지만 모두 확정 사실과 맞음 (드물다). 8~9 = 사소한 어색함만. 6~7 = 독자가 알아챌 비약이나 "
        "어긋남 하나. 5 이하 = 확정 사실과 충돌하거나 사건이 근거 없이 일어남.",
        "아래 키만 가진 JSON 객체 하나를 한 줄로 출력한다. 각 목록 항목은 '어느 대목이 어떤 사실과 어긋나는지' 를 60자 이내 한국어로 "
        "적고, 문제가 없으면 빈 목록으로 둔다.",
        '{"score": 1~10 정수, "contradictions": ["..."], "unmotivated": ["..."], "continuity_ok": true, '
        '"beat_progress": true, "verdict": "한 문장 총평"}',
    ])


def _clamp_score(value: Any) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(max(1, min(10, round(number))))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            items.append(text[:MAX_ITEM_CHARS])
        if len(items) >= MAX_ITEMS:
            break
    return items


def _flag(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "예", "그렇다"}:
            return True
        if lowered in {"false", "no", "0", "아니오", "아니다"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def dialogue_prompt(prefix: str, *, section: str, section_index: int) -> str:
    """대사 화자 점검 전용 프롬프트 — 좁은 과제 하나만 시킨다.

    일반 검토의 체크리스트 한 줄("동물이 말하면 모순")로는 모델이 자기 글의 화자 오류를 잘 못 잡고, 같은 검토
    안에 배정표를 넣어도 판정이 오락가락했다 (09-03 실측: 개가 "하린아, 으르렁!" 이라고 말하는 대목을 한 번은 잡고
    한 번은 놓침). 대사 배정만 시키는 짧은 호출은 접두사 캐시 덕에 거의 출력 시간만 든다.
    """
    return "\n".join([
        prefix.rstrip(),
        f"[이번 장({section_index}장) 본문 — 점검 대상]",
        section.strip()[:6000],
        f"{DIALOGUE_ROLE_MARKER} 위 본문의 대사(큰따옴표·낫표 안의 문장)를 등장 순서대로 모두 나열하고 화자를 배정한다. "
        "화자는 대사 바로 앞뒤 문장의 주체다 — '\"…\" 콩떡이 소리를 냈다' 처럼 대사 뒤에 붙은 문장의 주체가 화자로 읽힌다. "
        "먼저 [인물]에서 사람 말을 할 수 없는 존재(동물·물건·기계·이미 죽었거나 이 장면에 없는 인물)를 골라 non_speakers 에 "
        "적고, 그 존재가 화자로 읽히는 대사는 can_speak 을 false 로 둔다. 화자를 알 수 없으면 speaker 는 '불명', can_speak 은 true.",
        "아래 키만 가진 JSON 객체 하나를 한 줄로 출력한다. 대사가 없으면 dialogue 는 빈 목록.",
        '{"non_speakers": ["이름"], "dialogue": [{"quote": "대사 첫 여덟 자", "speaker": "화자 이름 또는 불명", "can_speak": true}]}',
    ])


def attribute_dialogue(
    client: Any,
    *,
    prefix: str,
    section: str,
    section_index: int,
    system: str,
    max_tokens: int = 500,
) -> list[str]:
    """말할 수 없는 존재의 대사를 모순 문장 목록으로. 대사가 없거나 호출이 실패하면 빈 목록."""
    if not any(mark in section for mark in QUOTE_CHARS):
        return []
    try:
        raw = client.chat(dialogue_prompt(prefix, section=section, section_index=section_index), system=system,
                          temperature=0.0, max_tokens=int(max_tokens), json_mode=True)
    except Exception as exc:  # noqa: BLE001 - 보조 점검, 실패하면 일반 검토만 쓴다
        logger.warning("Dialogue attribution for section %d failed: %s", section_index, exc)
        return []
    payload = json_object(raw) or {}
    return _impossible_speakers(payload.get("dialogue"))


def _impossible_speakers(value: Any) -> list[str]:
    """대사 배정표에서 말할 수 없는 화자에게 붙은 대사를 모순 문장으로 바꾼다."""
    if not isinstance(value, list):
        return []
    found: list[str] = []
    for item in value:
        if not isinstance(item, dict) or _flag(item.get("can_speak"), default=True):
            continue
        quote = " ".join(str(item.get("quote") or "").split())[:24]
        speaker = " ".join(str(item.get("speaker") or "불명").split())[:20] or "불명"
        if quote:
            found.append(f"대사 '{quote}' 가 말할 수 없는 존재({speaker})의 말로 읽힘")
        if len(found) >= MAX_ITEMS:
            break
    return found


def parse_report(raw: str) -> PlausibilityReport | None:
    """모델 출력에서 검토 결과를 읽는다. 점수가 없으면 None (검토 불가로 처리)."""
    payload = json_object(raw)
    if not payload:
        return None
    score = _clamp_score(payload.get("score"))
    if score is None:
        return None
    contradictions = _string_list(payload.get("contradictions"))
    for issue in _impossible_speakers(payload.get("dialogue")):
        if issue not in contradictions and len(contradictions) < MAX_ITEMS:
            contradictions.append(issue)
    return PlausibilityReport(
        score=score,
        contradictions=contradictions,
        unmotivated=_string_list(payload.get("unmotivated")),
        continuity_ok=_flag(payload.get("continuity_ok")),
        beat_progress=_flag(payload.get("beat_progress")),
        verdict=" ".join(str(payload.get("verdict") or "").split())[:200],
    )


def review_section(
    client: Any,
    *,
    prefix: str,
    section: str,
    section_index: int,
    plan_text: str,
    instruction: str,
    system: str,
    max_tokens: int = 600,
) -> PlausibilityReport:
    """한 장을 검토한다: 대사 화자 점검(좁은 호출) + 연속성 검토(넓은 호출), 결과를 합친다.

    검토가 실패하면 available=False 인 보고서를 돌려 생성기가 그대로 진행하게 한다. 화자 점검만 실패하면
    연속성 검토 결과만 쓴다. 두 호출은 temperature 0 — 같은 장에 같은 판정이 나와야 재검토가 뜻이 있다.
    """
    speaker_issues = attribute_dialogue(client, prefix=prefix, section=section, section_index=section_index, system=system)
    prompt = review_prompt(prefix, section=section, section_index=section_index, plan_text=plan_text,
                           instruction=instruction)
    try:
        raw = client.chat(prompt, system=system, temperature=0.0, max_tokens=int(max_tokens), json_mode=True)
    except Exception as exc:  # noqa: BLE001 - 검토는 보조 단계, 본문을 잃지 않는다
        logger.warning("Plausibility review for section %d failed: %s", section_index, exc)
        return PlausibilityReport.unavailable(f"검토 호출 실패: {exc}")
    report = parse_report(raw)
    if report is None:
        logger.warning("Plausibility review for section %d returned no usable JSON (%d chars)",
                       section_index, len(raw or ""))
        return PlausibilityReport.unavailable("검토 응답을 읽을 수 없음")
    for issue in speaker_issues:
        # 같은 대사를 두고 두 호출이 다른 말로 지적할 수 있다 — 인용 구절이 이미 들어 있으면 한 번만 적는다.
        quote = issue.split("'")[1][:6] if issue.count("'") >= 2 else issue
        if any(quote in existing for existing in report.contradictions):
            continue
        if len(report.contradictions) < MAX_ITEMS:
            report.contradictions.append(issue)
    return report
