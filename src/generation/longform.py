"""장편 소설 생성기 — 로컬 Qwen3.5-4B 전용 (JEPA·임베딩 없이 구조적 장치만 유지).

설계 근거 (2026-08-23 실측, C:/연구_프로젝트/ai_아키텍처/applied/README.md):
- 4B 4bit 모델은 직전 장 전문을 문맥에 두면 그대로 베끼는 루프에 빠진다 → 앞 내용은
  요약·상태 원장(story_rag)으로 치환하고, 직전 장은 꼬리 일부만 원문으로 준다.
- 반복 페널티를 올리면 한자·영어로 샌다 → 페널티 1.05 + 한국어 토큰 필터(서버),
  대신 생성 후 길이·종결·복사·반복·이름 게이트로 최대 1회 재생성.
- 말투 LoRA 는 서술을 짧게 만든다 → 소설은 항상 기본 모델.

흐름: 이야기 지도(outline, 1회) → 장별 과제(primary function + 소비된 비트)
→ 장 생성 → 게이트 → 1회 재생성 → 메모리 기록(별도 짧은 호출) → 원자 저장.
워커와의 계약은 generate_with_controlled_hallucination 과 동일하다.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable

from src.generation.consistency import check_name_consistency, extract_character_names
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

TraceCallback = Callable[[str, str, dict[str, Any] | None], None]
GENERATOR_NAME = "longform-qwen"
SECTION_HEADING_RE = re.compile(r"(?m)^###\s+")
SENTENCE_END = ".!?。！？…\"'”’)]」』"
FOREIGN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0400-\u04ff]")
CREATIVITY_ANCHOR = 0.35

SYSTEM_PROMPT = (
    "당신은 한국어 장편 소설을 장(章) 단위로 이어 쓰는 작가다. 독자에게 보이는 본문만 쓴다. "
    "설정과 앞선 사건을 지키되 같은 장면을 되풀이하지 않고, 매 장마다 이야기를 한 걸음 앞으로 옮긴다."
)
MEMORY_SYSTEM_PROMPT = "당신은 소설 편집자다. 본문에서 확정된 사실만 간결한 JSON 으로 기록한다."


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
    if callback is not None and _has_stream_control(callback):
        getattr(callback, method)(*args)


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


def _section_prompt(
    *,
    world: str,
    characters: str,
    outline_text: str,
    memory_context: str,
    consumed_context: str,
    tail: str,
    section_index: int,
    section_role: str,
    function_name: str,
    function_rule: str,
    target_chars: int,
    completion_rule: str,
    instruction: str,
    revision_notes: list[str] | None = None,
) -> str:
    parts = [
        "[작품 설정]", world.strip(),
        "[인물 — 이 이름들만 쓴다]", characters.strip() or "(주인공 한 명)",
        "[이야기 지도]", outline_text.strip(),
        "[지금까지의 줄거리와 현재 상태]", memory_context.strip(),
        "[이미 지나간 사건 — 새 사실처럼 반복하지 말 것]", consumed_context.strip(),
        f"[직전 장면의 끝부분 — 여기서 바로 이어 쓴다]", tail.strip() or "(첫 장면이다)",
        f"[이번 장({section_index}장)의 과제]",
        f"- 역할: {section_role}",
        f"- 핵심 서사 기능 하나: {function_name} — {function_rule}",
        f"- 분량: 약 {target_chars:,}자 (공백 포함). 대화와 묘사를 섞고, 구체적인 행동·장소 변화·상태 변화를 하나 이상 넣는다.",
        f"- 마무리 규칙: {completion_rule}",
        f"- 사용자 요청: {instruction.strip() or '기존의 미해결 압력과 인물의 목표를 따라간다.'}",
        "[작성 규칙]",
        "- 첫 줄은 '### 소제목' 한 줄. 그 뒤는 본문 문단만 쓴다 (목록·해설·제목 반복·메모 금지).",
        "- 직전 장면을 요약하거나 되풀이하지 않고 바로 새 사건을 진행한다.",
        "- 인물표에 없는 새 이름을 만들지 않는다. 한국어로만 쓴다.",
        "- 마지막 문장을 완전히 끝맺는다.",
    ]
    if revision_notes:
        parts.extend(["[이전 시도의 문제 — 이번엔 반드시 피할 것]", *[f"- {note}" for note in revision_notes]])
    return "\n".join(part for part in parts if part is not None)


def _memory_prompt(section: str, section_index: int, known_names: list[str]) -> str:
    names = ", ".join(known_names) if known_names else "(인물표 없음)"
    return (
        f"다음은 한국어 장편 소설의 {section_index}장이다. 이 장에서 확정된 사실만 기록하라. 인물 이름은 {names} 중에서만 쓴다.\n\n"
        f"[본문]\n{section[:3500]}\n\n"
        "아래 키를 가진 JSON 객체 하나만, 줄바꿈 없이 한 줄로 출력하라. 각 목록은 최대 4개, 각 항목은 30자 이내 한국어.\n"
        '{"title": "소제목", "summary": "한 문장 요약", "characters": ["등장 인물"], "facts": ["새로 확정된 사실"], '
        '"open_clues": ["미해결 단서"], "resolved_clues": ["해결된 단서"], "locations": ["장소"], '
        '"state_changes": ["인물·물건·목표의 상태 변화"], '
        '"state_updates": [{"entity": "인물/물건", "attribute": "location|emotion|goal|status|owner", "value": "현재 값"}], '
        '"keywords": ["핵심어 3~6개"]}'
    )


def _salvage_json_object(text: str) -> dict[str, Any] | None:
    """토큰 상한에 잘린 JSON 객체를 마지막 완결 항목까지 살려 닫는다.

    괄호 스택을 추적하며 쉼표·닫는 괄호 위치를 '안전 지점'으로 모은 뒤, 뒤에서부터
    하나씩 잘라 닫아 보고 처음 파싱되는 후보를 돌려준다. 값 없이 끝난 키나 미완성
    문자열은 자연히 건너뛴다.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    try:
        payload = json.loads(body[: body.rfind("}") + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_string = False
    escape = False
    safe_points: list[tuple[int, list[str]]] = []
    for index, char in enumerate(body):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
            safe_points.append((index + 1, list(stack)))
        elif char == ",":
            safe_points.append((index, list(stack)))
    for cut, open_brackets in reversed(safe_points[-80:]):
        candidate = body[:cut].rstrip().rstrip(",")
        candidate = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", candidate).rstrip().rstrip(",")
        candidate += "".join(reversed(open_brackets))
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload:
            return _prune_incomplete_items(payload)
    return None


def _prune_incomplete_items(payload: dict[str, Any]) -> dict[str, Any]:
    """잘린 꼬리에서 살아남은 불완전 객체(키가 모자란 항목)를 목록에서 뺀다."""
    for key, value in list(payload.items()):
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            widest = max(len(item) for item in value)
            payload[key] = [item for item in value if len(item) == widest]
    return payload


def _extract_memory(client: Any, config: AppConfig, section: str, section_index: int, known_names: list[str]) -> StoryMemory:
    title = _section_title(section, section_index)
    try:
        raw = client.chat(
            _memory_prompt(section, section_index, known_names),
            system=MEMORY_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=int(config.llm.memory_max_tokens),
            json_mode=True,
            korean_filter=True,
        )
    except Exception:  # noqa: BLE001 - 메모리는 보조 데이터, 본문을 잃지 않는다
        raw = ""
    memory = _parse_memory_payload(raw, section_index, title) if raw else None
    if memory is None and raw:
        salvaged = _salvage_json_object(raw)
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
        "version": 2,
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
    payload["retry_reasons"] = list(payload.get("retry_reasons", []))[-20:]
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
    if config.generation.enable_consumed_beat_ledger:
        for index, (section, memory) in enumerate(zip(sections, memories, strict=True), start=1):
            consumed.extend(extract_consumed_beats(section, memory=memory, section_index=index, genre=genre))

    turns_completed = int(run_state.get("turns_completed", 0) or 0)
    current_turn = turns_completed + 1
    counters = {
        "repetition_retry_count": int(run_state.get("repetition_retry_count", 0) or 0),
        "retry_success_count": int(run_state.get("retry_success_count", 0) or 0),
        "stability_retry_count": int(run_state.get("stability_retry_count", 0) or 0),
        "stability_retry_success_count": int(run_state.get("stability_retry_success_count", 0) or 0),
    }
    start_counters = dict(counters)
    retry_reasons = [str(item) for item in (run_state.get("retry_reasons", []) or [])]
    stability_scores: list[float] = []
    memory_retrievals = 0
    turn_sections: list[str] = []
    turn_start_chars = len("\n\n".join(sections))
    total_chars = turn_start_chars
    turn_floor = int(turn_target * 0.9)
    temperature = _creativity_temperature(config)
    recent_chars = max(400, int(config.generation.longform_recent_context_chars))
    max_tokens = int(max(600, min(config.llm.novel_max_tokens, section_target / 1.5 + 250)))
    minimum_chars = int(section_target * float(config.generation.stability_min_section_ratio))
    _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=current_turn,
             turn_target_chars=turn_target, novel_completed=False, retry_reasons=retry_reasons, **counters)
    _emit(trace_callback, "장편 생성 준비", "done", {
        "turn": current_turn, "sections": len(sections), "overall_target": overall_target,
        "turn_target": turn_target, "temperature": temperature, "max_tokens": max_tokens,
    })

    section_index = len(sections) + 1
    generated = 0
    ending_written = False
    while section_index <= max_total_sections and (
        (not ending_written and total_chars + section_target >= overall_target)
        or ((total_chars - turn_start_chars) < turn_floor and generated < turn_section_cap)
    ):
        previous_body = _section_body(sections[-1]) if sections else ""
        tail = previous_body[-recent_chars:] if previous_body else previous_scene.strip()[-recent_chars:]
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
        consumed_context = (
            build_consumed_beat_context(consumed, max_chars=max(300, int(config.generation.consumed_beat_context_chars)))
            if config.generation.enable_consumed_beat_ledger else "(없음)"
        )
        query = "\n".join([outline_text, continuation_instruction, tail])
        retrieved = (
            retrieve_story_memories(memories, query, config.generation.story_memory_top_k, section_index)
            if config.generation.enable_story_memory_rag else []
        )
        memory_retrievals += len(retrieved)
        memory_context, _ledger = format_hierarchical_story_context(
            memories, retrieved, query, max(400, int(config.generation.story_memory_context_chars)),
            group_size=config.generation.story_summary_group_size,
        )
        prompt_kwargs = dict(
            world=world, characters=characters, outline_text=outline_text, memory_context=memory_context,
            consumed_context=consumed_context, tail=tail, section_index=section_index, section_role=section_role,
            function_name=function_name, function_rule=function_rule, target_chars=section_target,
            completion_rule=completion_rule, instruction=continuation_instruction,
        )
        _emit(trace_callback, "장 생성", "running", {
            "section": section_index, "turn": current_turn, "turn_section": generated + 1,
            "chars": total_chars, "outline_beat": beat_index + 1, "function": function_name,
            "memory_hits": len(retrieved),
        })
        _stream(stream_callback, "begin_section", "\n\n" if turn_sections else "")
        started = time.monotonic()
        try:
            candidate = _generate_section(
                client, config, _section_prompt(**prompt_kwargs), temperature, max_tokens, section_index,
                stream_callback, seed=None,
            )
            check = assess_section(
                candidate, previous_body=previous_body, characters=characters, prior_titles=titles,
                minimum_chars=minimum_chars, consumed_beats=consumed, memory=None, genre=genre,
            )
            if check.hard:  # 연한 문제만으로는 재생성하지 않는다 (재생성 = 장 하나 분량의 시간)
                counters["stability_retry_count"] += 1
                if any("되풀이" in item or "반복" in item for item in check.hard):
                    counters["repetition_retry_count"] += 1
                retry_reasons.extend(f"{section_index}장: {reason}" for reason in check.issues)
                _emit(trace_callback, "장 재생성", "running", {"section": section_index, "reasons": check.issues})
                _stream(stream_callback, "restart_section", "이 장을 한 번 더 다듬는 중이야...")
                retry = _generate_section(
                    client, config,
                    _section_prompt(**prompt_kwargs, revision_notes=check.issues),
                    min(1.1, temperature + 0.1), max_tokens, section_index, stream_callback, seed=section_index * 7919,
                )
                retry_check = assess_section(
                    retry, previous_body=previous_body, characters=characters, prior_titles=titles,
                    minimum_chars=minimum_chars, consumed_beats=consumed, memory=None, genre=genre,
                )
                better = bool(retry) and (
                    (len(retry_check.hard), len(retry_check.issues)) < (len(check.hard), len(check.issues))
                )
                if better:
                    candidate, check = retry, retry_check
                    if not retry_check.hard:
                        counters["stability_retry_success_count"] += 1
                        if counters["repetition_retry_count"] > start_counters["repetition_retry_count"]:
                            counters["retry_success_count"] += 1
                _emit(trace_callback, "장 재생성", "done", {"section": section_index, "used": better, "remaining": check.issues})
            if not candidate:
                raise RuntimeError("모델이 빈 장을 돌려줬어.")
            memory = _extract_memory(client, config, candidate, section_index, known_names)
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
        if config.generation.enable_consumed_beat_ledger:
            consumed.extend(extract_consumed_beats(candidate, memory=memory, section_index=section_index, genre=genre))
        stability_scores.append(check.score)
        total_chars = len("\n\n".join(sections))
        generated += 1
        ending_written = ending_written or final_section
        _persist(config, sections, memories, turns_completed=turns_completed, turn_in_progress=current_turn,
                 turn_target_chars=turn_target, novel_completed=ending_written, retry_reasons=retry_reasons, **counters)
        _emit(trace_callback, "장 생성", "done", {
            "section": section_index, "chars": len(candidate), "total_chars": total_chars,
            "issues": check.issues, "score": check.score, "seconds": round(time.monotonic() - started, 1),
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
        "turn_repetition_retries": counters["repetition_retry_count"] - start_counters["repetition_retry_count"],
        "turn_retry_successes": counters["retry_success_count"] - start_counters["retry_success_count"],
        "turn_stability_retries": counters["stability_retry_count"] - start_counters["stability_retry_count"],
        "turn_stability_retry_successes": counters["stability_retry_success_count"] - start_counters["stability_retry_success_count"],
        "mean_stability_score": mean_stability,
        "mean_jepa_coherence": 0.0,
        "min_jepa_coherence": 0.0,
        "turn_coherence_retries": 0,
        "turn_coherence_retry_successes": 0,
        "jepa_coherence_by_section": {},
        "retry_reasons": retry_reasons[-20:],
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
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        stream_callback=visible,
        korean_filter="strict",  # 소설 본문: 한자·가나 + 영어 단어 토큰까지 차단 (08-23 실측 누출 대응)
        **options,
    )
    section = _normalize_section(raw, section_index)
    if not section:
        return ""
    body = _trim_to_last_sentence(_section_body(section))
    return f"### {_section_title(section, section_index)}\n\n{body}"
