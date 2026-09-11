"""장편 소설 생성기 — Ollama Gemma4 26B-A4B (기본) / 레거시 Qwen3.5-4B 모델 서버.

설계 근거:
- 랩 실측(2026-08-23, 4B, C:/연구_프로젝트/ai_아키텍처/applied/README.md): 직전 장 전문을 문맥에 두면 베끼는
  루프 → 앞 내용은 요약·상태 원장(story_rag)으로 치환하고 직전 장은 꼬리만 원문으로 준다. 반복 페널티를 올리면
  언어 이탈 → 페널티 1.05 + 생성 후 게이트. 26B 로 바꾼 뒤에도 이 구조는 그대로 둔다.
- 개연성(2026-09-03): 장을 쓰기 전에 인과 설계(목표·장애·전환·결과)를 정하고, 쓴 뒤 연속성 편집자 역할의 모델이
  확정 사실·앞선 사건·직전 장면과의 모순과 원인 없는 사건을 검토한다(plausibility). 모순이 있거나 점수가 기준
  미만이면 문제 목록을 들고 한 번 고쳐 쓰고 다시 검토한다.
- 모든 장 단위 호출(설계·생성·검토·기록)은 같은 시스템 프롬프트 + 같은 순서의 접두사(작품 설정 → 인물 →
  이야기 지도 → 줄거리·상태 → 지나간 사건 → 직전 장면 끝 → 집필 지침)로 시작한다. Ollama 가 접두사 KV 캐시를
  재사용해 뒤따르는 호출의 프리필이 거의 0 이 된다 (09-03 실측 8.7K 토큰 16.4s → 0.4s).

흐름: 이야기 지도(1회) → 장 설계 → 장 생성 → 기계 게이트(길이·종결·복사·반복·이름; 반복은 걷어내기, 그 외 1회
재생성) → 개연성 검토 → (문제 시) 고쳐 쓰기 → 재검토 → 메모리 기록 → 원자 저장.
두 종류의 다시 쓰기는 다르다. 기계 게이트의 재생성은 버린 초안을 프롬프트에 넣지 않는다 (복사 유인 차단,
tests/test_longform_retry_isolation.py). 개연성 고쳐 쓰기는 초안을 넣고 문제만 바로잡게 한다 — 사건 순서와
문장을 지키며 모순을 고치는 데는 그게 맞다 (tests/test_plausibility_gate.py).
워커와의 계약은 generate_with_controlled_hallucination 과 동일하다.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable

from src.generation.consistency import check_name_consistency, extract_character_names
from src.generation.plausibility import PlausibilityReport, review_section
from src.llm.jsonish import json_object, salvage_json_object
from src.memory.beat_ledger import (
    ConsumedBeat,
    build_consumed_beat_context,
    choose_primary_function,
    extract_consumed_beats,
    likely_repeats_consumed_beat,
)
from src.memory.story_outline import (
    StoryOutline,
    create_story_outline,
    load_story_outline,
    outline_context,
    write_story_outline,
)
from src.memory.story_rag import (
    StoryMemory,
    _fallback_memory,
    _parse_memory_payload,
    format_hierarchical_story_context,
    load_story_memories,
    retrieve_story_memories,
    strip_machine_block,
    write_story_ledger,
    write_story_memories,
)
from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path

logger = logging.getLogger(__name__)

TraceCallback = Callable[[str, str, dict[str, Any] | None], None]
GENERATOR_NAME = "longform"
SECTION_HEADING_RE = re.compile(r"(?m)^###\s+")
SENTENCE_END = ".!?。！？…\"'”’)]」』"
FOREIGN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0400-\u04ff]")
CREATIVITY_ANCHOR = 0.35
# 문맥 예산: 한국어 1.5자/토큰 실측(Gemma4)보다 보수적으로 잡는다. 접두사 뒤에 붙는 가장 긴 과제(검토 = 장 본문
# + 지시)를 위해 남겨 두는 글자 수.
CHARS_PER_TOKEN = 1.45
TASK_RESERVE_CHARS = 3600
LOCAL_SERVER_CONTEXT_TOKENS = 110_000
MIN_MEMORY_CHARS = 600
MIN_CONSUMED_CHARS = 300
MIN_TAIL_CHARS = 600

# 모든 장 단위 호출이 공유하는 시스템 프롬프트. 역할은 사용자 메시지 끝에서 정한다 (접두사 캐시 유지).
NOVEL_SYSTEM_PROMPT = (
    "당신은 한국어 장편 소설 제작팀이다. 요청에 따라 작가·장 설계자·연속성 편집자·기록 담당 역할로 답한다. "
    "어느 역할이든 [작품 설정]·[인물]·[지금까지의 줄거리와 현재 상태]에 적힌 확정 사실을 어기지 않고, "
    "지나간 사건을 새 사실처럼 되풀이하지 않으며, 모든 사건에 앞선 사건이나 장면 안의 원인을 둔다. "
    "작가 역할일 때는 독자에게 보이는 본문만 쓴다."
)
SYSTEM_PROMPT = NOVEL_SYSTEM_PROMPT
MEMORY_SYSTEM_PROMPT = NOVEL_SYSTEM_PROMPT
PLAN_ROLE_MARKER = "[역할: 장 설계자]"
WRITE_ROLE_MARKER = "[작성 규칙 — 작가 역할]"
REPAIR_MARKER = "[역할: 작가 — 검토 결과 고쳐 쓰기]"
MEMORY_ROLE_MARKER = "[역할: 기록 담당]"
PLAN_BLOCK_HEADER = "[이번 장의 설계 — 이 인과를 따른다]"

# story_editor 가 예전 이름으로 가져다 쓴다.
_salvage_json_object = salvage_json_object


def _emit(trace: TraceCallback | None, stage: str, status: str, detail: dict[str, Any] | None = None) -> None:
    if trace is not None:
        trace(stage, status, detail)


# ---- 스트림 제어 프로토콜 (story_workspace.LiveProseWriter 와 동일) ----------------------

def _has_stream_control(callback: Callable[[str], None] | None) -> bool:
    return bool(
        callback is not None
        and all(callable(getattr(callback, name, None)) for name in
                ("begin_section", "restart_section", "commit_section", "abort_section"))
    )


def _stream(callback: Callable[[str], None] | None, method: str, *args: Any) -> None:
    """스트림 제어 호출. note_section 처럼 선택적인 메서드는 콜백에 없으면 건너뛴다."""
    if callback is not None and _has_stream_control(callback):
        handler = getattr(callback, method, None)
        if callable(handler):
            handler(*args)


def _revise_stream(callback: Callable[[str], None] | None, reason: str) -> None:
    """개연성 고쳐 쓰기 시작: 화면의 초안을 치우고 사유를 남긴다. revise_section 이 없는 콜백은 restart 로 대신한다."""
    if callback is None or not _has_stream_control(callback):
        return
    handler = getattr(callback, "revise_section", None)
    if callable(handler):
        handler(reason)
    else:
        callback.restart_section(reason)  # type: ignore[attr-defined]


# ---- 텍스트 유틸 ---------------------------------------------------------------------------

def _sections_from_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [_normalize_section(text, 1)]
    return [
        text[m.start(): matches[i + 1].start() if i + 1 < len(matches) else None].strip()
        for i, m in enumerate(matches)
    ]


def _section_title(section: str, index: int) -> str:
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            return stripped[4:].strip()[:80]
    return f"장면 {index}"


def _section_body(section: str) -> str:
    lines = section.strip().splitlines()
    if lines and lines[0].strip().startswith("### "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _normalize_section(raw: str, index: int) -> str:
    """모델 출력에서 소제목 하나 + 본문만 남긴다. 마크다운 제목이 우선, '제목:' 줄은 폴백."""
    text = strip_machine_block(raw).strip()
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    heading_title = ""
    label_title = ""
    body: list[str] = []
    for line in [line.rstrip() for line in text.splitlines()]:
        stripped = line.strip()
        heading = re.match(r"^#{1,6}\s*(.+)$", stripped)
        if heading:
            if not heading_title:
                heading_title = re.sub(r"^(소제목|제목|장 제목)\s*[:：]\s*", "", heading.group(1).strip().strip("*").strip())[:80]
            continue  # 두 번째 이후 소제목은 버린다 (한 장 = 한 소제목)
        label = re.match(r"^(제목|소제목|장 제목)\s*[:：]\s*(.*)$", stripped)
        if label:
            if not label_title:
                label_title = label.group(2).strip()[:80]
            continue
        # 본문 앞의 "제 2 장: 붉은 비의 신호" / "2장. 제목" 같은 장 표기도 소제목으로 쓴다 (08-23 실측).
        chapter = re.match(r"^(?:제\s*\d+\s*장|\d+\s*장)\s*[:：.\-]?\s*(.{0,60})$", stripped)
        if chapter and not body and len(stripped) <= 70:
            if not label_title:
                label_title = chapter.group(1).strip() or stripped
            continue
        body.append(line)
    body_text = re.sub(r"\n{3,}", "\n\n", "\n".join(body).strip())
    if not body_text:
        return ""
    title = heading_title or label_title or f"장면 {index}"
    return f"### {title}\n\n{body_text}"


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+")
REPEAT_KEY_CHARS = 12  # 이보다 짧은 문장("응.", "그래.")은 정당하게 되풀이될 수 있어 세지 않는다
TRIMMABLE_ISSUES = ("같은 구절", "마지막 문장")  # 걷어내기로 풀 수 있는 심각 문제


def trim_repetitions(section: str) -> tuple[str, int]:
    """되풀이된 문단·문장을 첫 등장만 남기고 걷어낸다. (정리된 장, 걷어낸 개수).

    반복 루프는 대개 같은 문장이나 문단을 그대로 다시 쓰는 형태라, 장 전체를 다시 생성하기 전에 기계적으로
    걷어내면 전개를 지킨 채 통과시킬 수 있다. 문장 안에 같은 구절이 박힌 경우는 못 잡는다 — 그때는 게이트가
    그대로 남아 재생성으로 간다.
    """
    title = _section_title(section, 0)
    body = _section_body(section)
    removed = 0
    seen_paragraphs: set[str] = set()
    seen_sentences: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        if not paragraph.strip():
            continue
        paragraph_key = "".join(paragraph.split())
        if paragraph_key in seen_paragraphs:
            removed += 1
            continue
        seen_paragraphs.add(paragraph_key)
        lines: list[str] = []
        for line in paragraph.splitlines():
            kept: list[str] = []
            for sentence in SENTENCE_SPLIT_RE.split(line.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                key = "".join(sentence.split())
                if len(key) >= REPEAT_KEY_CHARS and key in seen_sentences:
                    removed += 1
                    continue
                seen_sentences.add(key)
                kept.append(sentence)
            if kept:
                lines.append(" ".join(kept))
        if lines:
            paragraphs.append("\n".join(lines))
    cleaned = "\n\n".join(paragraphs).strip()
    if not cleaned:
        return "", removed
    return f"### {title}\n\n{cleaned}", removed


def _trim_to_last_sentence(body: str) -> str:
    """상한에 걸려 잘린 꼬리를 마지막 완결 문장까지 되돌린다 (버리는 꼬리는 300자 또는 40% 이내)."""
    if not body or body[-1] in SENTENCE_END:
        return body
    cut = max(body.rfind(ch) for ch in ".!?…。")
    if cut >= 0 and (len(body) - cut - 1) <= max(300, int(len(body) * 0.4)):
        return body[: cut + 1].rstrip()
    return body


def _max_repeated_span(text: str, width: int = 12) -> int:
    hangul = re.sub(r"\s+", "", text)
    if len(hangul) < width * 2:
        return 0
    counts = Counter(hangul[i : i + width] for i in range(len(hangul) - width + 1))
    return counts.most_common(1)[0][1] if counts else 0


def _distinct3(text: str) -> float:
    hangul = "".join(ch for ch in text if "가" <= ch <= "힣")
    grams = [hangul[i : i + 3] for i in range(max(0, len(hangul) - 2))]
    return len(set(grams)) / max(1, len(grams))


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left[-3000:], right[:3000]).ratio()


# ---- 게이트 ----------------------------------------------------------------------------------

class SectionCheck:
    def __init__(self) -> None:
        self.hard: list[str] = []
        self.soft: list[str] = []

    @property
    def issues(self) -> list[str]:
        return [*self.hard, *self.soft]

    @property
    def score(self) -> float:
        return max(0.0, round(1.0 - len(self.hard) * 0.3 - len(self.soft) * 0.12, 4))


def assess_section(
    section: str,
    *,
    previous_body: str,
    characters: str,
    prior_titles: list[str],
    minimum_chars: int,
    consumed_beats: list[ConsumedBeat],
    memory: StoryMemory | None,
    genre: str,
) -> SectionCheck:
    check = SectionCheck()
    body = _section_body(section)
    title = _section_title(section, 0)
    if len(body) < max(200, minimum_chars):
        check.hard.append(f"본문이 너무 짧음 ({len(body)}자 < {minimum_chars}자)")
    if body and body[-1] not in SENTENCE_END:
        check.hard.append("마지막 문장이 끝맺어지지 않음")
    similarity = _similarity(previous_body, body)
    if similarity >= 0.55:  # 실측: 실제 복사 0.94~1.00, 준복사 0.62~0.83, 정상 0.06~0.46
        check.hard.append(f"직전 장을 거의 그대로 되풀이함 (유사도 {similarity:.2f})")
    repeated_span = _max_repeated_span(body)
    if repeated_span >= 6:
        check.hard.append(f"같은 구절이 {repeated_span}회 반복됨")
    elif repeated_span >= 4:
        check.soft.append(f"같은 구절이 {repeated_span}회 반복됨")
    elif _distinct3(body) < 0.5 and len(body) > 600:
        check.soft.append("문장 다양성이 낮음")
    if title and title.strip() in {item.strip() for item in prior_titles if item.strip()}:
        check.soft.append(f"소제목이 앞 장과 겹침: {title}")
    names = check_name_consistency(section, characters)
    check.soft.extend(names.issues[:3])
    foreign = len(FOREIGN_RE.findall(body))
    if foreign:
        check.soft.append(f"외국 문자 {foreign}자 혼입")
    if consumed_beats:
        repeated, reasons = likely_repeats_consumed_beat(section, consumed_beats, memory=memory, genre=genre)
        if repeated:
            check.soft.extend(reasons[:2])
    return check


# ---- 프롬프트 ------------------------------------------------------------------------------

def _completion_rule(total_chars: int, overall_target: int, section_target: int, *, final_of_turn: bool) -> tuple[str, bool]:
    projected = total_chars + section_target
    if projected >= overall_target:
        return (
            "이 장이 소설의 마지막 장이다. 중심 갈등을 해결하고 주인공의 결정적 선택과 그 결과를 보여 준 뒤, "
            "여운이 남는 마지막 장면으로 끝낸다. 새 수수께끼나 다음 장 예고를 넣지 않는다.",
            True,
        )
    if projected >= int(overall_target * 0.9):
        return "절정을 향해 밀어붙이고 정리할 수 있는 곁가지는 정리하되, 중심 갈등의 해결은 마지막 장을 위해 남겨 둔다.", False
    if final_of_turn:
        return "이 장의 끝은 다음 장이 궁금해지는 장면으로 맺되, 문장이나 행동을 중간에 끊지 않는다.", False
    return "중심 갈등을 아직 해결하지 않고, 앞으로 나아가는 압력을 남긴 채 끝낸다.", False


def _creativity_temperature(config: AppConfig) -> float:
    base = float(config.llm.novel_temperature)
    target = max(0.0, min(1.0, float(config.generation.hallucination_target)))
    return max(0.3, min(1.1, base + (target - CREATIVITY_ANCHOR) * 0.6))


def section_max_tokens(config: AppConfig, section_target: int) -> int:
    """장 하나의 생성 토큰 상한: 목표 글자 수 / 1.5 + 여유, 설정 상한 안에서."""
    return int(max(600, min(int(config.llm.novel_max_tokens), section_target / 1.5 + 250)))


def prompt_char_budget(config: AppConfig, max_tokens: int) -> int:
    """프롬프트 전체가 넘지 말아야 할 글자 수. Ollama 는 num_ctx 를 넘는 앞부분(작품 설정·인물)을 소리 없이 자른다."""
    if str(config.llm.backend).strip().lower() == "local":
        window = LOCAL_SERVER_CONTEXT_TOKENS
    else:
        window = int(config.llm.num_ctx)
    return max(2000, int((window - int(max_tokens) - 600) * CHARS_PER_TOKEN))


def _fit_prefix(
    build: Callable[[int, int, int], str],
    *,
    budget: int,
    memory_chars: int,
    consumed_chars: int,
    tail_chars: int,
) -> str:
    """접두사를 문맥 예산에 맞춘다: 요약 메모리 → 지나간 사건 → 직전 장 꼬리 순으로 줄인다 (꼬리는 끝을 지킨다)."""
    limit = max(800, budget - TASK_RESERVE_CHARS)
    prefix = build(memory_chars, consumed_chars, tail_chars)
    while len(prefix) > limit:
        if memory_chars > MIN_MEMORY_CHARS:
            memory_chars = max(MIN_MEMORY_CHARS, int(memory_chars * 0.7))
        elif consumed_chars > MIN_CONSUMED_CHARS:
            consumed_chars = max(MIN_CONSUMED_CHARS, int(consumed_chars * 0.7))
        elif tail_chars > MIN_TAIL_CHARS:
            tail_chars = max(MIN_TAIL_CHARS, int(tail_chars * 0.7))
        else:
            break
        prefix = build(memory_chars, consumed_chars, tail_chars)
    return prefix


def _canon_prefix(
    *,
    world: str,
    characters: str,
    outline_text: str,
    memory_context: str,
    consumed_context: str,
    tail: str,
    style_guide: str = "",
) -> str:
    """설계·생성·검토·기록 호출이 공유하는 접두사. 순서를 바꾸거나 앞에 무언가를 끼우면 Ollama 접두사 캐시가 깨진다."""
    parts = [
        "[작품 설정]", world.strip(),
        "[인물 — 이 이름들만 쓴다]", characters.strip() or "(주인공 한 명)",
        "[이야기 지도]", outline_text.strip(),
        "[지금까지의 줄거리와 현재 상태]", memory_context.strip(),
        "[이미 지나간 사건 — 새 사실처럼 반복하지 말 것]", consumed_context.strip(),
        "[직전 장면의 끝부분 — 여기서 바로 이어 쓴다]", tail.strip() or "(첫 장면이다)",
    ]
    if style_guide.strip():
        parts.extend(["[집필 지침 — 작가가 정한 문체·시점·금기]", style_guide.strip()])
    return "\n".join(parts) + "\n"


def _task_block(
    *,
    section_index: int,
    section_role: str,
    function_name: str,
    function_rule: str,
    target_chars: int,
    completion_rule: str,
    instruction: str,
) -> str:
    return "\n".join([
        f"[이번 장({section_index}장)의 과제]",
        f"- 역할: {section_role}",
        f"- 핵심 서사 기능 하나: {function_name} — {function_rule}",
        f"- 분량: 약 {target_chars:,}자 (공백 포함). 대화와 묘사를 섞고, 구체적인 행동·장소 변화·상태 변화를 하나 이상 넣는다.",
        f"- 마무리 규칙: {completion_rule}",
        f"- 사용자 요청: {instruction.strip() or '기존의 미해결 압력과 인물의 목표를 따라간다.'}",
    ])


def _section_prompt(prefix: str, task: str, *, plan_text: str = "", revision_notes: list[str] | None = None) -> str:
    parts = [prefix.rstrip(), task]
    if plan_text.strip():
        parts.extend([PLAN_BLOCK_HEADER, plan_text.strip()])
    parts.extend([
        WRITE_ROLE_MARKER,
        "- 첫 줄은 '### 소제목' 한 줄. 그 뒤는 본문 문단만 쓴다 (목록·해설·제목 반복·메모 금지).",
        "- 직전 장면을 요약하거나 되풀이하지 않고 바로 새 사건을 진행한다.",
        "- 모든 사건에는 앞선 사건이나 이 장 안의 원인이 있어야 한다. 확정된 사실(장소·소유·관계·목표·생사)을 바꿀 때는 "
        "그 원인을 장면으로 보여 준다. 우연으로 문제를 풀지 않는다.",
        "- 인물표에 없는 새 이름을 만들지 않는다. 한국어로만 쓴다.",
        "- 마지막 문장을 완전히 끝맺는다.",
    ])
    if revision_notes:
        parts.extend(["[이전 시도의 문제 — 이번엔 반드시 피할 것]", *[f"- {note}" for note in revision_notes]])
    return "\n".join(parts)


def _plan_prompt(prefix: str, task: str) -> str:
    return "\n".join([
        prefix.rstrip(),
        task,
        f"{PLAN_ROLE_MARKER} 본문을 쓰기 전에 이 장의 인과 설계를 정한다. [지금까지의 줄거리와 현재 상태]의 확정 사실과 "
        "미해결 단서, [직전 장면의 끝부분]의 상황에서 출발해 위 과제(핵심 서사 기능·사용자 요청·마무리 규칙)를 이루는 "
        "사건 하나를 설계한다. 새 인물·새 설정을 만들지 않고, [이미 지나간 사건]을 다시 겪게 하지 않는다. "
        "모든 항목은 '무엇 때문에 무엇이 일어나는지' 가 보이게 쓴다.",
        "아래 키만 가진 JSON 객체 하나를 한 줄로 출력한다. 각 값은 60자 이내 한국어.",
        '{"goal": "시점 인물이 이 장에서 이루려는 것", "obstacle": "그것을 막는 것 (앞선 사건에서 비롯)", '
        '"turn": "장 중반의 전환 — 새 정보·선택·사건", "outcome": "장이 끝날 때 달라진 상태와 그 원인", '
        '"uses": ["가져다 쓰는 앞선 단서·사실 1~3개"], "avoid": ["되풀이하면 안 되는 지나간 사건 1~2개"]}',
    ])


def _string_items(value: Any, limit: int = 3) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = [" ".join(str(item).split())[:120] for item in value if str(item).strip()]
    return items[:limit]


def _render_plan(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, label in (("goal", "목표"), ("obstacle", "장애"), ("turn", "전환"), ("outcome", "결과")):
        value = " ".join(str(payload.get(key) or "").split())[:160]
        if value:
            lines.append(f"- {label}: {value}")
    uses = _string_items(payload.get("uses"))
    avoid = _string_items(payload.get("avoid"))
    if uses:
        lines.append("- 가져다 쓰는 앞선 사실: " + "; ".join(uses))
    if avoid:
        lines.append("- 되풀이 금지: " + "; ".join(avoid))
    return "\n".join(lines) if len(lines) >= 2 else ""


def _plan_section(client: Any, config: AppConfig, prefix: str, task: str, section_index: int) -> str:
    """장을 쓰기 전에 인과 설계를 받는다. 실패하면 빈 문자열 — 설계 없이 쓴다."""
    try:
        raw = client.chat(
            _plan_prompt(prefix, task),
            system=NOVEL_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=int(config.llm.plan_max_tokens),
            json_mode=True,
        )
    except Exception as exc:  # noqa: BLE001 - 설계는 보조 단계, 없이도 쓴다
        logger.warning("Scene plan for section %d failed: %s", section_index, exc)
        return ""
    plan = _render_plan(json_object(raw) or {})
    if not plan:
        logger.warning("Scene plan for section %d returned no usable JSON (%d chars)", section_index, len(raw or ""))
    return plan


def _repair_prompt(prefix: str, *, section: str, section_index: int, issues: list[str], plan_text: str) -> str:
    parts = [
        prefix.rstrip(),
        f"[이번 장({section_index}장) 본문 — 고칠 대상]",
        section.strip(),
        "[검토에서 나온 문제 — 모두 고친다]",
        *[f"- {issue}" for issue in issues],
    ]
    if plan_text.strip():
        parts.extend([PLAN_BLOCK_HEADER, plan_text.strip()])
    parts.append(
        f"{REPAIR_MARKER} 위 문제만 바로잡아 이 장 전체를 다시 쓴다. 모순은 [작품 설정]·[인물]·[지금까지의 줄거리와 현재 "
        "상태]의 확정 사실에 맞게 고치고, 원인 없는 사건에는 앞선 사건에서 비롯한 원인을 장면으로 넣거나 그 사건을 뺀다. "
        "문제와 무관한 문장·사건 순서·분량·소제목은 그대로 둔다. 첫 줄은 '### 소제목', 그 뒤는 본문 문단만. 인물표에 "
        "없는 새 이름을 만들지 않고 한국어로만 쓰며 마지막 문장을 완전히 끝맺는다."
    )
    return "\n".join(parts)


def _memory_prompt(section: str, section_index: int, known_names: list[str], *, prefix: str = "") -> str:
    names = ", ".join(known_names) if known_names else "(인물표 없음)"
    if prefix:
        head = [prefix.rstrip(), f"[이번 장({section_index}장) 본문 — 기록 대상]"]
    else:
        head = [f"다음은 한국어 장편 소설의 {section_index}장이다.", "[본문]"]
    return "\n".join([
        *head,
        section[:3500],
        f"{MEMORY_ROLE_MARKER} 이 장에서 새로 확정된 사실만 기록한다. 인물 이름은 {names} 중에서만 쓴다. 앞선 장에서 "
        "이미 확정된 사실은 다시 적지 않는다. state_updates 에는 이 장에서 실제로 바뀐 값만(location|emotion|goal|status|owner), "
        "open_clues 에는 이 장이 남긴 미해결 질문·약속·위협, resolved_clues 에는 이 장에서 답이 난 앞선 단서를 적는다.",
        "아래 키를 가진 JSON 객체 하나만, 줄바꿈 없이 한 줄로 출력하라. 각 목록은 최대 4개, 각 항목은 40자 이내 한국어.",
        '{"title": "소제목", "summary": "한 문장 요약", "characters": ["등장 인물"], "facts": ["새로 확정된 사실"], '
        '"open_clues": ["미해결 단서"], "resolved_clues": ["해결된 단서"], "locations": ["장소"], '
        '"state_changes": ["인물·물건·목표의 상태 변화"], '
        '"state_updates": [{"entity": "인물/물건", "attribute": "location|emotion|goal|status|owner", "value": "현재 값"}], '
        '"relations": [{"source": "인물/물건", "relation": "possesses|trusts|hides|seeks|located_at|causes", "target": "대상"}], '
        '"keywords": ["핵심어 3~6개"]}',
    ])


def _extract_memory(
    client: Any,
    config: AppConfig,
    section: str,
    section_index: int,
    known_names: list[str],
    *,
    prefix: str = "",
) -> StoryMemory:
    title = _section_title(section, section_index)
    try:
        raw = client.chat(
            _memory_prompt(section, section_index, known_names, prefix=prefix),
            system=NOVEL_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=int(config.llm.memory_max_tokens),
            json_mode=True,
            korean_filter=True,
        )
    except Exception:  # noqa: BLE001 - 메모리는 보조 데이터, 본문을 잃지 않는다
        raw = ""
    memory = _parse_memory_payload(raw, section_index, title) if raw else None
    if memory is None and raw:
        salvaged = salvage_json_object(raw)
        if salvaged:
            salvaged.setdefault("section_index", section_index)
            salvaged.setdefault("title", title)
            try:
                memory = StoryMemory.model_validate(salvaged)
            except ValueError:
                memory = None
    if memory is None:
        memory = _fallback_memory(section, section_index, title, known_names)
    memory.section_index = section_index
    memory.title = (memory.title or title).strip()[:80]
    for field in ("characters", "facts", "open_clues", "resolved_clues", "locations", "state_changes", "keywords"):
        setattr(memory, field, [str(item)[:160] for item in getattr(memory, field)][:8])
    memory.state_updates = memory.state_updates[:8]
    memory.relations = memory.relations[:8]
    if known_names:
        memory.characters = [name for name in memory.characters if any(k in name or name in k for k in known_names)] or [
            name for name in known_names if name in section
        ]
    return memory


# ---- 상태 저장 -------------------------------------------------------------------------------

def _write_checkpoint(config: AppConfig, sections: list[str]) -> str:
    path = resolve_path(config, config.generation.longform_checkpoint_path)
    ensure_parent(path)
    path.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")
    return str(path)


def _load_run_state(config: AppConfig) -> dict[str, Any]:
    path = resolve_path(config, config.generation.longform_state_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_run_state(config: AppConfig, sections: list[str], memories: list[StoryMemory], **values: Any) -> str:
    path = resolve_path(config, config.generation.longform_state_path)
    ensure_parent(path)
    text = "\n\n".join(sections).strip()
    payload = {
        "version": 3,
        "generator": GENERATOR_NAME,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_chars": len(text),
        "section_count": len(sections),
        "memory_count": len(memories),
        "checkpoint_path": str(resolve_path(config, config.generation.longform_checkpoint_path)),
        "memory_path": str(resolve_path(config, config.generation.story_memory_path)),
        "ledger_path": str(resolve_path(config, config.generation.story_ledger_path)),
    }
    payload.update(values)
    payload["retry_reasons"] = list(payload.get("retry_reasons", []))[-30:]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _persist(config: AppConfig, sections: list[str], memories: list[StoryMemory], **state: Any) -> dict[str, str]:
    return {
        "checkpoint": _write_checkpoint(config, sections),
        "memory": write_story_memories(resolve_path(config, config.generation.story_memory_path), memories),
        "ledger": write_story_ledger(
            resolve_path(config, config.generation.story_ledger_path), memories,
            group_size=config.generation.story_summary_group_size,
        ),
        "state": _write_run_state(config, sections, memories, **state),
    }


# ---- 메인 --------------------------------------------------------------------------------------

COUNTER_KEYS = (
    "repetition_trim_count",
    "repetition_retry_count",
    "retry_success_count",
    "stability_retry_count",
    "stability_retry_success_count",
    "plausibility_repair_count",
    "plausibility_repair_success_count",
)


def generate_longform(
    config: AppConfig,
    client: Any,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callback: Callable[[str], None] | None = None,
    scene_preset: dict[str, str] | None = None,
    return_details: bool = False,
    trace_callback: TraceCallback | None = None,
    continue_existing: bool = False,
    turn_target_chars: int | None = None,
    continuation_instruction: str = "",
    style_guide: str = "",
) -> str | dict[str, Any]:
    del scene_preset  # 레거시 인터페이스 호환
    draft_path = resolve_path(config, config.generation.longform_checkpoint_path)
    sections = _sections_from_text(draft_path.read_text(encoding="utf-8")) if (continue_existing and draft_path.exists()) else []
    if continue_existing and not sections:
        raise FileNotFoundError("이어 쓸 원고가 없어. 먼저 첫 장을 시작해줘.")
    run_state = _load_run_state(config) if continue_existing else {}
    if continue_existing and bool(run_state.get("novel_completed", False)):
        raise RuntimeError("이 이야기는 이미 결말까지 완성됐어. 전체 원고를 내려받거나 새 이야기를 시작해줘.")

    known_names = extract_character_names(characters)
    genre = str(config.generation.genre or "")
    memories = load_story_memories(resolve_path(config, config.generation.story_memory_path)) if continue_existing else []
    if len(memories) < len(sections):
        memories = memories + [
            _fallback_memory(section, index, _section_title(section, index), known_names)
            for index, section in enumerate(sections[len(memories):], start=len(memories) + 1)
        ]
    memories = memories[: len(sections)]

    overall_target = max(1000, int(config.generation.target_novel_chars))
    section_target = max(600, int(config.generation.section_min_chars))
    turn_target = max(600, int(turn_target_chars or config.generation.turn_target_chars))
    planned_sections = max(int(config.generation.section_count), math.ceil(overall_target / section_target), 1)
    max_total_sections = max(planned_sections, int(config.generation.longform_max_sections))
    if len(sections) >= max_total_sections:
        raise RuntimeError(f"원고가 설정된 최대 장 수({max_total_sections})에 도달했어.")
    expected_turn_sections = max(1, math.ceil(turn_target / section_target))
    turn_section_cap = min(max(1, int(config.generation.turn_max_sections)), max_total_sections - len(sections),
                           expected_turn_sections + 1)
    if getattr(client, "dry_run", False):
        turn_section_cap = min(turn_section_cap, 2)

    outline: StoryOutline | None = None
    outline_path = resolve_path(config, config.generation.story_outline_path)
    if config.generation.enable_story_outline:
        outline = load_story_outline(outline_path)
        if outline is None:
            _emit(trace_callback, "이야기 지도 설계", "running", None)
            outline = create_story_outline(
                client, world=world, characters=characters, premise=previous_scene,
                target_chars=overall_target, beat_count=config.generation.outline_beat_count,
            )
            write_story_outline(outline_path, outline)
            _emit(trace_callback, "이야기 지도 설계", "done", {"beats": len(outline.beats)})

    story_seed = f"{genre}\n{world.strip()}\n{characters.strip()}"
    titles = [_section_title(section, index) for index, section in enumerate(sections, start=1)]
    consumed: list[ConsumedBeat] = []
    enable_consumed = bool(config.generation.enable_consumed_beat_ledger)
    if enable_consumed:
        for index, (section, memory) in enumerate(zip(sections, memories, strict=True), start=1):
            consumed.extend(extract_consumed_beats(section, memory=memory, section_index=index, genre=genre))

    turns_completed = int(run_state.get("turns_completed", 0) or 0)
    current_turn = turns_completed + 1
    counters = {key: int(run_state.get(key, 0) or 0) for key in COUNTER_KEYS}
    start_counters = dict(counters)
    retry_reasons = [str(item) for item in (run_state.get("retry_reasons", []) or [])]
    turn_reason_start = len(retry_reasons)
    stability_scores: list[float] = []
    plausibility_scores: dict[int, int] = {}
    memory_retrievals = 0
    turn_sections: list[str] = []
    turn_start_chars = len("\n\n".join(sections))
    total_chars = turn_start_chars
    turn_floor = int(turn_target * 0.9)
    temperature = _creativity_temperature(config)
    recent_chars = max(400, int(config.generation.longform_recent_context_chars))
    memory_budget = max(400, int(config.generation.story_memory_context_chars))
    consumed_budget = max(300, int(config.generation.consumed_beat_context_chars))
    max_tokens = section_max_tokens(config, section_target)
    char_budget = prompt_char_budget(config, max_tokens)
    minimum_chars = int(section_target * float(config.generation.stability_min_section_ratio))
    min_score = int(config.generation.plausibility_min_score)
    enable_plan = bool(config.generation.enable_scene_plan)
    enable_review = bool(config.generation.enable_plausibility_gate)
    _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=current_turn,
             turn_target_chars=turn_target, novel_completed=False, retry_reasons=retry_reasons, **counters)
    _emit(trace_callback, "장편 생성 준비", "done", {
        "turn": current_turn, "sections": len(sections), "overall_target": overall_target,
        "turn_target": turn_target, "temperature": temperature, "max_tokens": max_tokens, "char_budget": char_budget,
    })

    section_index = len(sections) + 1
    generated = 0
    ending_written = False
    while section_index <= max_total_sections and (
        (not ending_written and total_chars + section_target >= overall_target)
        or ((total_chars - turn_start_chars) < turn_floor and generated < turn_section_cap)
    ):
        previous_body = _section_body(sections[-1]) if sections else ""
        tail_source = previous_body if previous_body else previous_scene.strip()
        final_of_turn = generated + 1 >= expected_turn_sections
        completion_rule, final_section = _completion_rule(total_chars, overall_target, section_target, final_of_turn=final_of_turn)
        if outline is not None:
            outline_text, beat_index = outline_context(outline, section_index=section_index, planned_sections=planned_sections)
            section_role = ("이야기 지도의 현재 beat 를 하나의 구체적인 장면으로 극화한다: 감각적 디테일, "
                            "행동과 대사로 드러나는 갈등, 시점 인물의 관찰 가능한 반응.")
        else:
            outline_text, beat_index = "(이야기 지도 없음)", -1
            section_role = "미해결 갈등을 새로운 결과와 인물의 의미 있는 결정으로 한 단계 전진시킨다."
        if final_section:
            section_role = "중심 갈등을 해결하고 결정적 결과를 보여 준 뒤 마지막 이미지로 소설을 끝맺는다."
        function_name, function_rule = choose_primary_function(section_index, consumed, story_seed=story_seed)
        query = "\n".join([outline_text, continuation_instruction, tail_source[-recent_chars:]])
        retrieved = (
            retrieve_story_memories(memories, query, config.generation.story_memory_top_k, section_index)
            if config.generation.enable_story_memory_rag else []
        )
        memory_retrievals += len(retrieved)

        def build_prefix(memory_chars: int, consumed_chars: int, tail_chars: int) -> str:
            memory_context, _ledger = format_hierarchical_story_context(
                memories, retrieved, query, max(400, memory_chars), group_size=config.generation.story_summary_group_size,
            )
            consumed_context = (
                build_consumed_beat_context(consumed, max_chars=max(200, consumed_chars)) if enable_consumed else "(없음)"
            )
            return _canon_prefix(
                world=world, characters=characters, outline_text=outline_text, memory_context=memory_context,
                consumed_context=consumed_context, tail=tail_source[-tail_chars:] if tail_source else "",
                style_guide=style_guide,
            )

        prefix = _fit_prefix(build_prefix, budget=char_budget, memory_chars=memory_budget,
                             consumed_chars=consumed_budget, tail_chars=recent_chars)
        task = _task_block(
            section_index=section_index, section_role=section_role, function_name=function_name,
            function_rule=function_rule, target_chars=section_target, completion_rule=completion_rule,
            instruction=continuation_instruction,
        )
        _emit(trace_callback, "장 생성", "running", {
            "section": section_index, "turn": current_turn, "turn_section": generated + 1,
            "chars": total_chars, "outline_beat": beat_index + 1, "function": function_name,
            "memory_hits": len(retrieved), "prefix_chars": len(prefix),
        })
        _stream(stream_callback, "begin_section", "\n\n" if turn_sections else "")
        started = time.monotonic()
        try:
            plan_text = _plan_section(client, config, prefix, task, section_index) if enable_plan else ""
            if enable_plan:
                _emit(trace_callback, "장 설계", "done", {"section": section_index, "planned": bool(plan_text)})
            candidate = _generate_section(
                client, config, _section_prompt(prefix, task, plan_text=plan_text), temperature, max_tokens,
                section_index, stream_callback, seed=None,
            )
            gate_kwargs = dict(previous_body=previous_body, characters=characters, prior_titles=titles,
                               minimum_chars=minimum_chars, consumed_beats=consumed, memory=None, genre=genre)
            check = assess_section(candidate, **gate_kwargs)
            if check.hard and all(issue.startswith(TRIMMABLE_ISSUES) for issue in check.hard) and any(
                issue.startswith("같은 구절") for issue in check.hard
            ):
                # 반복 루프만 문제면 되풀이된 문장을 걷어내고 다시 검사한다 — 전개를 지키고 재생성 시간을 아낀다.
                # 루프가 토큰 상한까지 돌면 마지막 문장이 끊기기 마련이라(08-24 실측: "66회 반복 + 미완"), 그 조합도
                # 걷어낸 뒤 마지막 완결 문장까지 되돌려 본다.
                trimmed, removed = trim_repetitions(candidate)
                if trimmed and removed:
                    trimmed = "### " + _section_title(trimmed, section_index) + "\n\n" + _trim_to_last_sentence(_section_body(trimmed))
                    trimmed_check = assess_section(trimmed, **gate_kwargs)
                    if not trimmed_check.hard:
                        counters["repetition_trim_count"] += 1
                        retry_reasons.append(f"{section_index}장: 반복 문장 {removed}개를 걷어냄 (다시 쓰지 않음)")
                        _emit(trace_callback, "반복 정리", "done", {"section": section_index, "removed": removed})
                        _stream(stream_callback, "note_section", "trim",
                                f"반복된 문장 {removed}개를 걷어내고 이어가.", "")
                        candidate, check = trimmed, trimmed_check
            if check.hard and config.generation.enable_stability_retry:
                # 연한 문제만으로는 재생성하지 않는다 (재생성 = 장 하나 분량의 시간).
                # 재생성 프롬프트는 첫 시도와 같은 접두사·과제·설계와 문제 목록만 쓴다 — 버려진 초안의 본문은
                # 어디에도 들어가지 않는다 (tests/test_longform_retry_isolation.py 가 고정).
                counters["stability_retry_count"] += 1
                if any("되풀이" in item or "반복" in item for item in check.hard):
                    counters["repetition_retry_count"] += 1
                retry_reasons.extend(f"{section_index}장: {reason}" for reason in check.issues)
                first_issues = "; ".join(check.hard)
                _emit(trace_callback, "장 재생성", "running", {"section": section_index, "reasons": check.issues})
                _stream(stream_callback, "restart_section", first_issues)
                retry = _generate_section(
                    client, config,
                    _section_prompt(prefix, task, plan_text=plan_text, revision_notes=check.issues),
                    min(1.1, temperature + 0.1), max_tokens, section_index, stream_callback, seed=section_index * 7919,
                )
                retry_check = assess_section(retry, **gate_kwargs)
                better = bool(retry) and (
                    (len(retry_check.hard), len(retry_check.issues)) < (len(check.hard), len(check.issues))
                )
                if better:
                    discarded = candidate
                    candidate, check = retry, retry_check
                    if not retry_check.hard:
                        counters["stability_retry_success_count"] += 1
                        if counters["repetition_retry_count"] > start_counters["repetition_retry_count"]:
                            counters["retry_success_count"] += 1
                    decision = f"다시 쓴 판을 채택했어. (처음 초안의 문제: {first_issues})"
                else:
                    discarded = retry
                    decision = "다시 쓴 판이 더 낫지 않아 처음 초안을 그대로 채택했어."
                _stream(stream_callback, "note_section", "decision", decision, discarded)
                _emit(trace_callback, "장 재생성", "done", {"section": section_index, "used": better, "remaining": check.issues})
            if not candidate:
                raise RuntimeError("모델이 빈 장을 돌려줬어.")

            # ---- 개연성 검토 → 고쳐 쓰기 → 재검토 ------------------------------------------------
            report: PlausibilityReport | None = None
            if enable_review:
                report = review_section(
                    client, prefix=prefix, section=candidate, section_index=section_index, plan_text=plan_text,
                    instruction=continuation_instruction, system=NOVEL_SYSTEM_PROMPT,
                    max_tokens=int(config.llm.review_max_tokens),
                )
                if report.available and not report.passes(min_score):
                    issues = report.issues(min_score)
                    counters["plausibility_repair_count"] += 1
                    retry_reasons.extend(f"{section_index}장: {issue}" for issue in issues)
                    _emit(trace_callback, "개연성 고쳐 쓰기", "running",
                          {"section": section_index, "score": report.score, "issues": issues})
                    _revise_stream(stream_callback, ("개연성 검토: " + "; ".join(issues))[:300])
                    repaired = _generate_section(
                        client, config,
                        _repair_prompt(prefix, section=candidate, section_index=section_index, issues=issues, plan_text=plan_text),
                        min(temperature, 0.6), max_tokens, section_index, stream_callback, seed=None,
                    )
                    repaired_check = assess_section(repaired, **gate_kwargs) if repaired else None
                    repaired_report: PlausibilityReport | None = None
                    if repaired and repaired_check is not None and not repaired_check.hard:
                        repaired_report = review_section(
                            client, prefix=prefix, section=repaired, section_index=section_index, plan_text=plan_text,
                            instruction=continuation_instruction, system=NOVEL_SYSTEM_PROMPT,
                            max_tokens=int(config.llm.review_max_tokens),
                        )
                    better = bool(
                        repaired_report is not None and repaired_report.available and (
                            len(repaired_report.issues(min_score)) < len(issues) or repaired_report.score > report.score
                        )
                    )
                    if better and repaired_check is not None and repaired_report is not None:
                        discarded = candidate
                        candidate, check, report = repaired, repaired_check, repaired_report
                        if report.passes(min_score):
                            counters["plausibility_repair_success_count"] += 1
                        decision = f"개연성 검토에서 나온 문제를 고친 판을 채택했어. (문제: {'; '.join(issues)[:200]})"
                    else:
                        discarded = repaired
                        if repaired_check is not None and repaired_check.hard:
                            remaining = "고친 판이 기계 게이트에 걸림: " + "; ".join(repaired_check.hard)
                        elif repaired_report is not None and repaired_report.available:
                            remaining = "남은 문제: " + ("; ".join(repaired_report.issues(min_score)) or repaired_report.verdict)
                        else:
                            remaining = "고친 판을 검토할 수 없었어"
                        decision = f"고친 판이 더 낫지 않아 처음 판을 그대로 채택했어. ({remaining[:200]})"
                    _stream(stream_callback, "note_section", "decision", decision, discarded)
                    _emit(trace_callback, "개연성 고쳐 쓰기", "done",
                          {"section": section_index, "used": better, "score": report.score if report.available else None})
                if report.available:
                    plausibility_scores[section_index] = int(report.score)
                retry_reasons.append(f"{section_index}장: {report.summary()}")
                _emit(trace_callback, "개연성 검토", "done", {
                    "section": section_index, "available": report.available, "score": report.score,
                    "notes": report.notes(),
                })
            memory = _extract_memory(client, config, candidate, section_index, known_names, prefix=prefix)
        except Exception as exc:
            _stream(stream_callback, "abort_section")
            paths = _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=current_turn,
                             turn_target_chars=turn_target, novel_completed=ending_written,
                             retry_reasons=retry_reasons, **counters)
            raise RuntimeError(
                f"{section_index}장에서 생성이 멈췄어. 원고는 {paths['checkpoint']} 에 저장됐어. {exc}"
            ) from exc

        _stream(stream_callback, "commit_section")
        sections.append(candidate)
        turn_sections.append(candidate)
        titles.append(_section_title(candidate, section_index))
        memories.append(memory)
        if enable_consumed:
            consumed.extend(extract_consumed_beats(candidate, memory=memory, section_index=section_index, genre=genre))
        stability_scores.append(check.score)
        total_chars = len("\n\n".join(sections))
        generated += 1
        ending_written = ending_written or final_section
        _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=current_turn,
                 turn_target_chars=turn_target, novel_completed=ending_written, retry_reasons=retry_reasons, **counters)
        _emit(trace_callback, "장 생성", "done", {
            "section": section_index, "chars": len(candidate), "total_chars": total_chars,
            "issues": check.issues, "score": check.score, "plausibility": plausibility_scores.get(section_index),
            "seconds": round(time.monotonic() - started, 1),
        })
        if final_section:
            break
        section_index += 1

    full_text = "\n\n".join(sections).strip()
    turn_text = "\n\n".join(turn_sections).strip()
    if turn_sections:
        turns_completed = current_turn
    paths = _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=None,
                     turn_target_chars=turn_target, novel_completed=ending_written, retry_reasons=retry_reasons, **counters)
    if not return_details:
        return full_text
    mean_stability = round(sum(stability_scores) / len(stability_scores), 4) if stability_scores else 0.0
    scores = list(plausibility_scores.values())
    details = {
        "generator": GENERATOR_NAME,
        "direction": "",
        "retrieval_mean_score": 0.0,
        "story_memory_retrievals": memory_retrievals,
        "story_memory_count": len(memories),
        "story_outline_beats": len(outline.beats) if outline else 0,
        "turn_number": current_turn,
        "turn_sections": len(turn_sections),
        "turn_actual_chars": len(turn_text),
        "turn_target_chars": turn_target,
        "actual_novel_chars": len(full_text),
        "target_novel_chars": overall_target,
        "completed_sections": len(sections),
        "novel_completed": ending_written,
        "turn_repetition_trims": counters["repetition_trim_count"] - start_counters["repetition_trim_count"],
        "turn_retry_reasons": retry_reasons[turn_reason_start:],
        "turn_repetition_retries": counters["repetition_retry_count"] - start_counters["repetition_retry_count"],
        "turn_retry_successes": counters["retry_success_count"] - start_counters["retry_success_count"],
        "turn_stability_retries": counters["stability_retry_count"] - start_counters["stability_retry_count"],
        "turn_stability_retry_successes": counters["stability_retry_success_count"] - start_counters["stability_retry_success_count"],
        "mean_stability_score": mean_stability,
        # 개연성 검토 (연속성 편집자 역할의 모델, 1~10). 검토가 불가능했던 장은 빠진다.
        "plausibility_by_section": dict(plausibility_scores),
        "mean_plausibility": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_plausibility": min(scores) if scores else 0,
        "turn_plausibility_repairs": counters["plausibility_repair_count"] - start_counters["plausibility_repair_count"],
        "turn_plausibility_repair_successes": (
            counters["plausibility_repair_success_count"] - start_counters["plausibility_repair_success_count"]
        ),
        # 레거시 JEPA 게이트 필드 (워커 지표 호환)
        "mean_jepa_coherence": 0.0,
        "min_jepa_coherence": 0.0,
        "turn_coherence_retries": 0,
        "turn_coherence_retry_successes": 0,
        "jepa_coherence_by_section": {},
        "retry_reasons": retry_reasons[-30:],
        "checkpoint_path": paths["checkpoint"],
        "story_memory_path": paths["memory"],
        "story_ledger_path": paths["ledger"],
        "run_state_path": paths["state"],
        "temperature": temperature,
    }
    return {"text": full_text, "turn_text": turn_text, "planner": details}


def _generate_section(
    client: Any,
    config: AppConfig,
    prompt: str,
    temperature: float,
    max_tokens: int,
    section_index: int,
    stream_callback: Callable[[str], None] | None,
    *,
    seed: int | None,
) -> str:
    visible = stream_callback if _has_stream_control(stream_callback) else None
    options: dict[str, Any] = {}
    if seed is not None:
        options["seed"] = seed
    llm = config.llm
    if float(getattr(llm, "novel_dry_multiplier", 0.0)) > 0:
        options.update(dry_multiplier=float(llm.novel_dry_multiplier), dry_base=float(llm.novel_dry_base),
                       dry_allowed_length=int(llm.novel_dry_allowed_length))
    if float(getattr(llm, "novel_min_p", 0.0)) > 0:
        options.update(min_p=float(llm.novel_min_p), top_p=1.0, top_k=0)
    raw = client.chat(
        prompt,
        system=NOVEL_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        stream_callback=visible,
        korean_filter="strict",  # 레거시 로컬 서버: 한자·가나 + 영어 단어 토큰까지 차단. Ollama 는 무시한다.
        **options,
    )
    section = _normalize_section(raw, section_index)
    if not section:
        return ""
    body = _trim_to_last_sentence(_section_body(section))
    return f"### {_section_title(section, section_index)}\n\n{body}"
