from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.memory.story_rag import StoryMemory


BeatType = Literal[
    "reveal",
    "alliance_shift",
    "system_warning",
    "location_move",
    "clue_resolution",
    "new_threat",
    "emotional_turn",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[가-힣]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
SECTION_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")

BEAT_TRIGGERS: dict[BeatType, tuple[str, ...]] = {
    "reveal": ("진실", "정체", "밝혀", "드러", "설계자", "피해자", "혈통", "비밀"),
    "alliance_shift": ("동맹", "협력", "손잡", "거래", "비즈니스", "신뢰", "약속", "배신"),
    "system_warning": ("시스템", "경고", "경보", "프로토콜", "인식", "정화", "알람", "오류"),
    "location_move": ("이동", "도착", "향해", "좌표", "핵심 서버", "연구동", "기지", "통로"),
    "clue_resolution": ("단서", "해독", "해결", "열쇠", "복원", "암호", "좌표", "증거"),
    "new_threat": ("위협", "추적", "공격", "봉쇄", "적", "침입", "폭발", "포위"),
    "emotional_turn": ("결심", "두려움", "분노", "죄책감", "믿음", "절망", "후회", "용서"),
}

PRIMARY_FUNCTIONS = (
    ("new_clue", "새 단서 하나를 구체적인 물건, 기록, 증언으로 얻는다."),
    ("relationship_change", "인물 관계가 행동과 선택의 결과로 한 단계 변한다."),
    ("location_movement", "새 장소로 실제 이동하며 환경과 위험 조건이 달라진다."),
    ("choice_consequence", "직전 선택의 대가가 물리적 사건이나 손실로 나타난다."),
    ("enemy_action", "적대 세력이나 시스템이 먼저 행동해 상황을 바꾼다."),
    ("clue_resolution", "기존 복선 하나를 회수하되 더 큰 결과를 남긴다."),
)


class ConsumedBeat(BaseModel):
    section_index: int
    beat_type: BeatType
    summary: str
    keywords: list[str] = Field(default_factory=list)


def extract_consumed_beats(
    section: str,
    memory: StoryMemory | None = None,
    section_index: int | None = None,
) -> list[ConsumedBeat]:
    index = section_index or (memory.section_index if memory is not None else 0)
    source_parts = [section]
    if memory is not None:
        source_parts.extend(
            [
                memory.summary,
                *memory.facts,
                *memory.open_clues,
                *memory.resolved_clues,
                *memory.state_changes,
                *(f"{item.entity} {item.attribute} {item.value}" for item in memory.state_updates),
                *(f"{item.source} {item.relation} {item.target}" for item in memory.relations),
            ]
        )
    source = "\n".join(part for part in source_parts if part)
    sentences = [item.strip() for item in SENTENCE_RE.split(source) if item.strip()]
    beats: list[ConsumedBeat] = []

    for beat_type, triggers in BEAT_TRIGGERS.items():
        matched = next(
            (
                sentence
                for sentence in sentences
                if any(trigger.lower() in sentence.lower() for trigger in triggers)
            ),
            "",
        )
        if not matched:
            continue
        keywords = _keywords(matched, triggers)
        beats.append(
            ConsumedBeat(
                section_index=index,
                beat_type=beat_type,
                summary=_clip_summary(matched),
                keywords=keywords,
            )
        )
    return beats


def build_consumed_beat_context(
    beats: list[ConsumedBeat],
    max_chars: int = 1200,
) -> str:
    if not beats:
        return "(none yet)"
    lines: list[str] = []
    for beat in sorted(beats, key=lambda item: item.section_index, reverse=True):
        line = f"- Section {beat.section_index} / {beat.beat_type}: {beat.summary}"
        candidate = "\n".join([*lines, line])
        if lines and len(candidate) > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)[:max_chars]


def likely_repeats_consumed_beat(
    section: str,
    beats: list[ConsumedBeat],
    *,
    memory: StoryMemory | None = None,
    previous_section: str = "",
) -> tuple[bool, list[str]]:
    if not section.strip() or not beats:
        return False, []

    candidate_beats = extract_consumed_beats(section, memory=memory)
    reasons: list[str] = []
    for candidate in candidate_beats:
        recent_same_type = [
            beat
            for beat in beats
            if beat.beat_type == candidate.beat_type
        ][-5:]
        for consumed in recent_same_type:
            overlap = _keyword_overlap(candidate.keywords, consumed.keywords)
            summary_overlap = _token_jaccard(candidate.summary, consumed.summary)
            shared_keywords = set(candidate.keywords) & set(consumed.keywords)
            shared_triggers = shared_keywords & _distinctive_tokens(candidate.beat_type)
            shared_content = shared_keywords - _distinctive_tokens(candidate.beat_type)
            if (
                overlap >= 0.5
                or summary_overlap >= 0.42
                or (shared_triggers and shared_content)
            ):
                reasons.append(
                    f"{candidate.beat_type} repeats section {consumed.section_index}: {consumed.summary}"
                )
                break

    if previous_section:
        similarity = section_similarity(previous_section, section)
        if similarity >= 0.48:
            reasons.append(f"adjacent section token similarity is too high ({similarity:.2f})")
    return bool(reasons), reasons


def build_section_direction(
    base_direction: str,
    section_role: str,
    primary_function: str,
    consumed_context: str,
) -> str:
    return "\n".join(
        [
            f"Base JEPA direction: {base_direction}",
            f"Current section role: {section_role}",
            f"Primary narrative function: {primary_function}",
            "Treat consumed beats as prior canon, not as fresh discoveries.",
            "Advance from their consequence through one concrete action, state change, or specific new fact.",
            "[Consumed beat summary]",
            consumed_context,
        ]
    )


def choose_primary_function(section_index: int, beats: list[ConsumedBeat]) -> tuple[str, str]:
    recent_types = {beat.beat_type for beat in beats[-6:]}
    candidates = list(PRIMARY_FUNCTIONS)
    if "reveal" in recent_types:
        candidates = [item for item in candidates if item[0] != "new_clue"] + [
            item for item in candidates if item[0] == "new_clue"
        ]
    offset = max(0, section_index - 1) % len(candidates)
    return candidates[offset]


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return []
    result: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append((match.group(1).strip(), text[match.end() : end].strip()))
    return result


def narrative_beat_metrics(text: str) -> dict[str, float | int]:
    sections = split_sections(text)
    if not sections:
        return {
            "repeated_subtitle_count": 0,
            "repeated_narrative_beat_count": 0,
            "adjacent_section_similarity": 0.0,
            "max_adjacent_section_similarity": 0.0,
        }

    normalized_titles = [_normalize(title) for title, _body in sections]
    title_counts = Counter(normalized_titles)
    repeated_titles = sum(count - 1 for count in title_counts.values() if count > 1)
    consumed: list[ConsumedBeat] = []
    repeated_beats = 0
    similarities: list[float] = []
    previous = ""
    for section_index, (_title, body) in enumerate(sections, start=1):
        _repeats, reasons = likely_repeats_consumed_beat(
            body,
            consumed,
            previous_section=previous,
        )
        repeated_beats += sum(
            1 for reason in reasons if not reason.startswith("adjacent section")
        )
        consumed.extend(extract_consumed_beats(body, section_index=section_index))
        if previous:
            similarities.append(section_similarity(previous, body))
        previous = body
    return {
        "repeated_subtitle_count": repeated_titles,
        "repeated_narrative_beat_count": repeated_beats,
        "adjacent_section_similarity": round(sum(similarities) / max(1, len(similarities)), 4),
        "max_adjacent_section_similarity": round(max(similarities, default=0.0), 4),
    }


def section_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _keywords(text: str, triggers: tuple[str, ...]) -> list[str]:
    tokens = _tokens(text)
    trigger_tokens = [trigger.lower() for trigger in triggers if trigger.lower() in text.lower()]
    common = [
        token
        for token, _count in Counter(tokens).most_common(12)
        if len(token) >= 2 and token not in {"그리고", "그러나", "하지만", "그것", "그녀", "그는"}
    ]
    return list(dict.fromkeys([*trigger_tokens, *common]))[:10]


def _keyword_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _distinctive_tokens(beat_type: BeatType) -> set[str]:
    return {token.lower() for token in BEAT_TRIGGERS[beat_type] if len(token) >= 2}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _clip_summary(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]
