from __future__ import annotations

import hashlib
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

# The base table above is skewed toward SF/thriller vocabulary, so a court drama
# or a romance would barely trip the repetition guard. These tables add the
# genre's own words for the same seven beats; they extend the base rather than
# replace it, because out-of-genre words simply never appear in that prose.
# Keys match src.llm.scene_presets.GENRE_SCENE_PRESETS.
GENRE_BEAT_TRIGGERS: dict[str, dict[BeatType, tuple[str, ...]]] = {
    "한국형 판타지 미스터리": {
        "reveal": ("정체", "봉인", "예언", "원혼", "족보", "저주"),
        "alliance_shift": ("계약", "굿", "청탁", "상인", "대가", "맹세"),
        "system_warning": ("징조", "경고", "부적", "금기", "예언", "울림"),
        "location_move": ("장터", "폐가", "산길", "사당", "골목", "석문"),
        "clue_resolution": ("유물", "문양", "부적", "증언", "이름", "흔적"),
        "new_threat": ("저주", "원혼", "추격", "재앙", "균열", "그림자"),
        "emotional_turn": ("연민", "공포", "각오", "원망", "체념", "속죄"),
    },
    "궁중 판타지": {
        "reveal": ("혈통", "밀지", "역모", "출생", "친필", "진상"),
        "alliance_shift": ("충성", "당파", "혼담", "후견", "밀약", "등돌"),
        "system_warning": ("어명", "칙서", "탄핵", "전갈", "소문", "경고"),
        "location_move": ("입궐", "편전", "연회장", "동궁", "옥사", "행차"),
        "clue_resolution": ("서찰", "인장", "장부", "증언", "물증", "족보"),
        "new_threat": ("사약", "역모", "자객", "숙청", "유배", "포박"),
        "emotional_turn": ("굴욕", "결단", "분노", "충심", "환멸", "체념"),
    },
    "현대 오컬트": {
        "reveal": ("정체", "의식", "제물", "기록", "저주", "실체"),
        "alliance_shift": ("의뢰", "협조", "배교", "묵인", "동행", "결별"),
        "system_warning": ("징조", "경고", "금기", "규칙", "전조", "예감"),
        "location_move": ("현장", "지하", "폐건물", "성당", "골목", "봉인지"),
        "clue_resolution": ("문서", "사진", "증언", "표식", "이름", "기록"),
        "new_threat": ("빙의", "추적", "습격", "붕괴", "잠식", "침범"),
        "emotional_turn": ("공포", "확신", "죄책", "체념", "각성", "연민"),
    },
    "로맨스 스릴러": {
        "reveal": ("과거", "거짓말", "정체", "고백", "관계", "진심"),
        "alliance_shift": ("연인", "이별", "재회", "약속", "의심", "고백"),
        "system_warning": ("협박", "경고", "메시지", "소문", "감시", "전화"),
        "location_move": ("집", "호텔", "차", "공항", "골목", "병원"),
        "clue_resolution": ("사진", "문자", "통화", "일기", "영수증", "증거"),
        "new_threat": ("스토킹", "협박", "추적", "폭로", "위협", "감금"),
        "emotional_turn": ("설렘", "질투", "불신", "용서", "체념", "결심"),
    },
    "무협 정치극": {
        "reveal": ("정체", "사문", "혈서", "내막", "배후", "무공"),
        "alliance_shift": ("의형제", "문파", "배신", "밀약", "파문", "귀순"),
        "system_warning": ("전갈", "격문", "소문", "금령", "경고", "포고"),
        "location_move": ("객잔", "산문", "관문", "장원", "강가", "협곡"),
        "clue_resolution": ("서신", "비급", "패", "증언", "흔적", "장부"),
        "new_threat": ("자객", "매복", "포위", "독", "추격", "결투"),
        "emotional_turn": ("의리", "분노", "회한", "각오", "환멸", "용서"),
    },
    "디스토피아 성장물": {
        "reveal": ("진상", "실험", "명단", "출신", "조작", "기록"),
        "alliance_shift": ("동료", "밀고", "조직", "이탈", "포섭", "결별"),
        "system_warning": ("통제", "경보", "배급", "검열", "감시", "규정"),
        "location_move": ("구역", "수용소", "지하", "관문", "공장", "국경"),
        "clue_resolution": ("서류", "번호", "기록", "증언", "지도", "표식"),
        "new_threat": ("숙청", "단속", "추적", "봉쇄", "처형", "징집"),
        "emotional_turn": ("각성", "분노", "체념", "희망", "죄책", "결심"),
    },
    "해양 모험": {
        "reveal": ("항로", "해도", "정체", "선원", "난파", "유래"),
        "alliance_shift": ("선원", "반란", "합류", "이탈", "약속", "배신"),
        "system_warning": ("폭풍", "경보", "조류", "안개", "신호", "예보"),
        "location_move": ("항구", "갑판", "섬", "해협", "선창", "암초"),
        "clue_resolution": ("해도", "일지", "표류물", "증언", "좌표", "유물"),
        "new_threat": ("폭풍", "해적", "좌초", "침몰", "추격", "괴물"),
        "emotional_turn": ("두려움", "결의", "향수", "후회", "신뢰", "절망"),
    },
    "법정 미스터리": {
        "reveal": ("진범", "위증", "알리바이", "자백", "전력", "누락"),
        "alliance_shift": ("변호", "합의", "제보", "결별", "협조", "선임"),
        "system_warning": ("영장", "소환", "기각", "고발", "경고", "통보"),
        "location_move": ("법정", "구치소", "사무실", "현장", "검찰청", "복도"),
        "clue_resolution": ("증거", "진술", "기록", "감정서", "서류", "영상"),
        "new_threat": ("기소", "구속", "협박", "폭로", "해임", "보복"),
        "emotional_turn": ("확신", "회의", "분노", "죄책", "체념", "결심"),
    },
    "학원 이능 배틀": {
        "reveal": ("능력", "정체", "혈통", "실험", "등급", "각성"),
        "alliance_shift": ("동료", "라이벌", "팀", "배신", "합류", "결별"),
        "system_warning": ("교칙", "경보", "소집", "징계", "공지", "경고"),
        "location_move": ("교실", "훈련장", "기숙사", "옥상", "지하", "결계"),
        "clue_resolution": ("기록", "명부", "표식", "증언", "실험지", "단서"),
        "new_threat": ("습격", "폭주", "결투", "봉인", "추적", "침입"),
        "emotional_turn": ("각오", "열등감", "분노", "신뢰", "두려움", "성장"),
    },
    "가족 드라마 미스터리": {
        "reveal": ("출생", "유언", "과거", "친자", "사고", "진상"),
        "alliance_shift": ("화해", "절연", "재혼", "상속", "용서", "의절"),
        "system_warning": ("통보", "고지서", "소문", "전화", "경고", "진단"),
        "location_move": ("본가", "병원", "묘지", "이사", "고향", "거실"),
        "clue_resolution": ("사진첩", "편지", "일기", "통장", "증언", "유품"),
        "new_threat": ("소송", "폭로", "빚", "실종", "발병", "협박"),
        "emotional_turn": ("원망", "그리움", "죄책", "용서", "체념", "화해"),
    },
    "사이버펑크 누아르": {
        "reveal": ("배후", "데이터", "정체", "조작", "청부", "기록"),
        "alliance_shift": ("의뢰", "거래", "밀고", "손절", "포섭", "배신"),
        "system_warning": ("추적", "경보", "차단", "해킹", "감시", "경고"),
        "location_move": ("뒷골목", "슬럼", "타워", "클럽", "지하도", "옥상"),
        "clue_resolution": ("로그", "칩", "영상", "증언", "계좌", "코드"),
        "new_threat": ("청부", "추격", "제거", "폭파", "봉쇄", "침입"),
        "emotional_turn": ("냉소", "분노", "미련", "각오", "환멸", "연민"),
    },
    "역사 대체물": {
        "reveal": ("사초", "출신", "기록", "진상", "밀서", "계보"),
        "alliance_shift": ("연합", "동맹", "귀순", "배반", "혼인", "밀약"),
        "system_warning": ("포고", "격문", "전령", "소문", "경고", "칙명"),
        "location_move": ("성곽", "진영", "도성", "국경", "나루", "관아"),
        "clue_resolution": ("사초", "서신", "지도", "증언", "장부", "인장"),
        "new_threat": ("반란", "침공", "포위", "역병", "숙청", "암살"),
        "emotional_turn": ("결의", "회한", "분노", "충심", "환멸", "체념"),
    },
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


def triggers_for_genre(genre: str | None) -> dict[BeatType, tuple[str, ...]]:
    extra = GENRE_BEAT_TRIGGERS.get(str(genre or "").strip())
    if not extra:
        return BEAT_TRIGGERS
    return {
        beat_type: tuple(dict.fromkeys([*base, *extra.get(beat_type, ())]))
        for beat_type, base in BEAT_TRIGGERS.items()
    }


def extract_consumed_beats(
    section: str,
    memory: StoryMemory | None = None,
    section_index: int | None = None,
    genre: str | None = None,
) -> list[ConsumedBeat]:
    index = section_index or (memory.section_index if memory is not None else 0)
    trigger_table = triggers_for_genre(genre)
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

    for beat_type, triggers in trigger_table.items():
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
    genre: str | None = None,
) -> tuple[bool, list[str]]:
    if not section.strip() or not beats:
        return False, []

    candidate_beats = extract_consumed_beats(section, memory=memory, genre=genre)
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
            trigger_tokens = _distinctive_tokens(candidate.beat_type, genre=genre)
            shared_triggers = shared_keywords & trigger_tokens
            shared_content = shared_keywords - trigger_tokens
            if (
                (overlap >= 0.65 and len(shared_content) >= 2)
                or summary_overlap >= 0.55
                or (shared_triggers and len(shared_content) >= 2)
            ):
                reasons.append(
                    f"{candidate.beat_type} repeats section {consumed.section_index}: {consumed.summary}"
                )
                break

    if previous_section:
        similarity = section_similarity(previous_section, section)
        if similarity >= 0.65:
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


def choose_primary_function(
    section_index: int,
    beats: list[ConsumedBeat],
    *,
    story_seed: str = "",
) -> tuple[str, str]:
    recent_types = {beat.beat_type for beat in beats[-6:]}
    candidates = list(PRIMARY_FUNCTIONS)
    if "reveal" in recent_types:
        candidates = [item for item in candidates if item[0] != "new_clue"] + [
            item for item in candidates if item[0] == "new_clue"
        ]
    # A bare section-index rotation gives every story the same function in the
    # same slot. Offsetting the phase by the story seed keeps the even spread
    # while letting different premises start the cycle at different points.
    phase = 0
    if story_seed:
        phase = int(hashlib.sha256(story_seed.encode("utf-8")).hexdigest()[:8], 16)
    offset = (max(0, section_index - 1) + phase) % len(candidates)
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


def _distinctive_tokens(beat_type: BeatType, genre: str | None = None) -> set[str]:
    # Genre words must count as triggers, not as shared content, or the guard
    # would treat ordinary genre vocabulary as evidence of a repeated beat.
    table = triggers_for_genre(genre)
    return {token.lower() for token in table[beat_type] if len(token) >= 2}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _clip_summary(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]
