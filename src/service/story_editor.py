"""생성된 원고의 장 단위 편집 — 직접 고치기, AI 퇴고, 요약 메모리 갱신.

장편 생성기는 앞 내용을 원문이 아니라 `memory.jsonl` 의 요약·사실·단서로 본다. 그래서 본문만 고치면
다음 장은 고치기 전 이야기를 이어 쓴다. 여기서는 장을 바꿀 때마다 그 장의 메모리를 모델로 다시 뽑고
(실패하면 본문에서 기계적으로), 원장(ledger)·이어쓰기 상태(state.json)·DB 진행 지표를 같이 맞춘다.

워커가 쓰는 동안에는 고치지 못한다 (ConsumerStore.get_editable_story 가 거부).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.generation.consistency import extract_character_names
from src.generation.longform import (
    _extract_memory,
    _normalize_section,
    _salvage_json_object,
    _section_body,
    _section_title,
)
from src.memory.story_rag import (
    StoryMemory,
    _fallback_memory,
    load_story_memories,
    write_story_ledger,
    write_story_memories,
)
from src.service.consumer_store import ConsumerStore
from src.service.story_sheets import character_sheet, style_guide, world_sheet
from src.service.story_workspace import StoryWorkspace, read_draft, split_sections
from src.utils.config import AppConfig

logger = logging.getLogger(__name__)
HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*\S")
# 빠른 요청 라벨 → 모델에게 주는 구체적 지시. 08-23 실측(2천 자 장, Qwen3.5-4B):
# - 라벨만 주면 원문을 거의 그대로 베끼거나(유사도 0.99) 문장을 '~으며, ~고' 로 이어 붙여 느슨하게 만든다.
# - "문장을 짧게 끊어라" 를 세게 주면 한 줄에 한 문장씩 토막 내고 분량을 4~6할로 깎는다.
# - "분량·문단 구조는 원문과 같게" 를 붙이면 다시 복사로 돌아가고, 대화 늘리기는 반복 루프에 빠진다.
#   → 고쳐 쓰기 전에 '실제로 바꿀 곳' 을 먼저 고르게 한다 (plan_edits). 지시문에는 분량을 적지 않는다.
REWRITE_DIRECTIVES = {
    "더 긴장감 있게": (
        "긴장감을 높인다: 원문의 문장을 지우지 말고, 문단마다 위협·시간 압박·불길한 낌새를 드러내는 문장을 하나씩 "
        "더하고, 인물의 내면 설명은 행동과 짧은 대사로 바꾼다."
    ),
    "문장을 짧고 담백하게": (
        "문장을 짧고 담백하게: 수식어와 비유를 덜어내고 한 문장에 한 가지만 말한다. "
        "문장을 '~으며', '~고' 로 잇지 않는다."
    ),
    "대화를 늘리고 설명을 줄여": (
        "대화를 늘린다: 서술로 풀어 놓은 감정과 정보를 인물의 대사로 옮기고(지우지 말고 대사로 바꾼다), "
        "문단마다 주고받는 대사를 한두 줄 더한다. 같은 말을 되풀이하는 대사는 쓰지 않는다."
    ),
    "감각 묘사를 풍부하게": (
        "감각 묘사를 풍부하게: 시각 말고도 소리·냄새·촉감·온도를 장면마다 하나 이상 넣고, "
        "추상적인 표현을 구체적인 사물과 동작으로 바꾼다."
    ),
    "오탈자와 어색한 문장만 고쳐": "오탈자·맞춤법·어색한 문장만 고친다. 그 밖의 문장은 원문 그대로 둔다.",
}
PLAN_SYSTEM_PROMPT = "당신은 한국어 소설 편집자다. 고칠 곳을 원문 구절로 구체적으로 짚는다. JSON 만 출력한다."
MAX_PLAN_EDITS = 10
QUICK_REWRITES = tuple(REWRITE_DIRECTIVES)
MINIMAL_EDIT_RE = re.compile(r"오탈자|맞춤법|어색한 문장만|최소한")
REWRITE_SYSTEM_PROMPT = (
    "당신은 한국어 장편 소설의 퇴고 편집자다. 주어진 장을 사용자의 요청대로 고쳐 쓰되, 그 장에서 일어나는 "
    "사건의 순서와 결과, 인물, 설정, 장이 끝날 때의 상황은 그대로 유지한다. 독자에게 보이는 본문만 쓴다."
)


class StoryEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class SectionView:
    index: int  # 1부터
    title: str
    body: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def label(self) -> str:
        return f"{self.index}장 · {self.title} · {self.chars:,}자"


# ---- 읽기 ------------------------------------------------------------------------------------

def list_sections(workspace: StoryWorkspace) -> list[SectionView]:
    sections = split_sections(read_draft(workspace.draft))
    return [
        SectionView(index=index, title=_section_title(section, index), body=_section_body(section), text=section)
        for index, section in enumerate(sections, start=1)
    ]


def section_memory(workspace: StoryWorkspace, index: int) -> StoryMemory | None:
    for memory in load_story_memories(workspace.memory):
        if int(memory.section_index) == int(index):
            return memory
    return None


# ---- 본문 정규화 -------------------------------------------------------------------------------

EMPTY_BODY_ERROR = "본문이 비어 있어. 장을 지우려면 아래 '대화'에서 그 턴의 ✕ 를 눌러줘."


def normalize_edit(text: str, index: int, fallback_title: str, *, from_model: bool = False) -> str:
    """본문을 '### 소제목 + 본문' 한 장으로 만든다.

    from_model=True: 모델 출력. 생성기의 정리기(_normalize_section)로 펜스·메모·'제목:' 라벨·장 표기를 걷어낸다.
    from_model=False: 사람이 적은 본문. 첫 줄의 '### 소제목' 만 읽고 나머지는 글자 그대로 둔다 —
    모델용 정리기는 '제목:' 이나 '#', 'N장' 으로 시작하는 본문 줄을 지워 버리므로 사람 글에는 쓰지 않는다.
    소제목 줄이 없으면 원래 소제목을 지킨다.
    """
    raw = str(text or "").replace("\r", "")
    if from_model:
        section = _normalize_section(raw, index)
        if not section or not _section_body(section).strip():
            raise StoryEditError(EMPTY_BODY_ERROR)
        title = _section_title(section, index)
        # _normalize_section 은 제목 줄('### ', '#제목', '소제목:', '2장. …')을 못 찾았을 때만 '장면 N' 을 쓴다.
        # 그때만 원래 소제목을 지킨다 — 모델이 일부러 '### 장면 N' 이라고 쓴 경우는 제외.
        if title == f"장면 {index}" and not HEADING_RE.search(raw):
            title = (fallback_title or title).strip()[:80] or title
        return f"### {title}\n\n{_section_body(section)}"
    lines = raw.strip().splitlines()
    title = (fallback_title or f"장면 {index}").strip()[:80]
    heading = re.match(r"^\s*#{1,6}\s*(.+?)\s*$", lines[0]) if lines else None
    if heading:
        title = heading.group(1).strip().strip("*").strip()[:80] or title
        lines = lines[1:]
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in lines).strip())
    if not body:
        raise StoryEditError(EMPTY_BODY_ERROR)
    return f"### {title}\n\n{body}"


def _write_draft(workspace: StoryWorkspace, sections: list[str]) -> int:
    text = "\n\n".join(sections).strip()
    workspace.draft.parent.mkdir(parents=True, exist_ok=True)
    workspace.draft.write_text(text + "\n", encoding="utf-8")
    return len(text)


def _aligned_memories(workspace: StoryWorkspace, sections: list[str], known_names: list[str]) -> list[StoryMemory]:
    """장 수와 메모리 수를 맞춘다 (모자라면 본문에서 기계적으로, 남으면 버림). 생성기와 같은 규칙."""
    memories = load_story_memories(workspace.memory)[: len(sections)]
    for index in range(len(memories) + 1, len(sections) + 1):
        section = sections[index - 1]
        memories.append(_fallback_memory(section, index, _section_title(section, index), known_names))
    for position, memory in enumerate(memories, start=1):
        memory.section_index = position
    return memories


def _write_memories(config: AppConfig, workspace: StoryWorkspace, memories: list[StoryMemory]) -> None:
    write_story_memories(workspace.memory, memories)
    write_story_ledger(workspace.ledger, memories, group_size=config.generation.story_summary_group_size)


def _known_names(story: dict[str, Any]) -> list[str]:
    return extract_character_names(character_sheet(story))


# ---- 쓰기 ------------------------------------------------------------------------------------

def replace_section(
    config: AppConfig,
    store: ConsumerStore,
    client: Any,
    owner_id: str,
    story_id: str,
    index: int,
    new_text: str,
    *,
    refresh_memory: bool = True,
) -> dict[str, Any]:
    """index 장의 본문을 바꾸고 메모리·원장·상태·진행 지표를 맞춘다.

    refresh_memory 가 참이면 그 장의 요약 메모리를 모델로 다시 뽑는다 (모델이 없거나 실패하면 본문에서
    기계적으로). 거짓이면 기존 메모리를 그대로 둔다.
    """
    story = store.get_editable_story(owner_id, story_id)
    workspace = StoryWorkspace.for_story(config, story_id, create=True)
    sections = split_sections(read_draft(workspace.draft))
    if not 1 <= int(index) <= len(sections):
        raise StoryEditError(f"{index}장은 없어. 지금 원고는 {len(sections)}장까지야.")
    position = int(index) - 1
    old = sections[position]
    section = normalize_edit(new_text, int(index), _section_title(old, int(index)))
    known_names = _known_names(story)
    changed = section != old
    memories = _aligned_memories(workspace, sections, known_names)
    if not changed and not refresh_memory:
        # 아무것도 바뀌지 않았으면 파일·DB 를 건드리지 않는다 (updated_at 도 그대로).
        return {"changed": False, "section": SectionView(index=int(index), title=_section_title(old, int(index)),
                                                          body=_section_body(old), text=old),
                "memory": memories[position], "total_chars": len("\n\n".join(sections).strip())}
    if changed:
        sections[position] = section
        logger.info("story %s: section %d replaced (%d -> %d chars)", story_id, index, len(old), len(section))
    total_chars = _write_draft(workspace, sections) if changed else len("\n\n".join(sections).strip())
    if refresh_memory:
        logger.info("story %s: re-extracting memory for section %d", story_id, index)
        memories[position] = _extract_memory(client, config, section, int(index), known_names)
    _write_memories(config, workspace, memories)
    store.sync_story_files(story_id, total_chars, len(sections), len(memories))
    return {
        "changed": changed,
        "section": SectionView(index=int(index), title=_section_title(section, int(index)),
                               body=_section_body(section), text=section),
        "memory": memories[position],
        "total_chars": total_chars,
    }


def refresh_section_memory(
    config: AppConfig, store: ConsumerStore, client: Any, owner_id: str, story_id: str, index: int
) -> StoryMemory:
    """본문은 그대로 두고 index 장의 요약 메모리만 모델로 다시 뽑는다."""
    story = store.get_editable_story(owner_id, story_id)
    workspace = StoryWorkspace.for_story(config, story_id, create=True)
    sections = split_sections(read_draft(workspace.draft))
    if not 1 <= int(index) <= len(sections):
        raise StoryEditError(f"{index}장은 없어. 지금 원고는 {len(sections)}장까지야.")
    known_names = _known_names(story)
    memories = _aligned_memories(workspace, sections, known_names)
    logger.info("story %s: re-extracting memory for section %d", story_id, index)
    memories[int(index) - 1] = _extract_memory(client, config, sections[int(index) - 1], int(index), known_names)
    _write_memories(config, workspace, memories)
    store.sync_story_files(story_id, len("\n\n".join(sections).strip()), len(sections), len(memories))
    return memories[int(index) - 1]


def _lines(value: str, limit: int = 8) -> list[str]:
    return [line.strip()[:160] for line in str(value or "").splitlines() if line.strip()][:limit]


def update_section_memory(
    config: AppConfig,
    store: ConsumerStore,
    owner_id: str,
    story_id: str,
    index: int,
    *,
    summary: str,
    facts: str = "",
    open_clues: str = "",
) -> StoryMemory:
    """사용자가 직접 적은 요약·사실·미해결 단서로 index 장의 메모리를 덮어쓴다 (줄마다 항목 하나)."""
    story = store.get_editable_story(owner_id, story_id)
    workspace = StoryWorkspace.for_story(config, story_id, create=True)
    sections = split_sections(read_draft(workspace.draft))
    if not 1 <= int(index) <= len(sections):
        raise StoryEditError(f"{index}장은 없어. 지금 원고는 {len(sections)}장까지야.")
    clean_summary = " ".join(str(summary or "").split())[:500]
    if not clean_summary:
        raise StoryEditError("요약은 비울 수 없어. 한 문장이라도 적어줘.")
    memories = _aligned_memories(workspace, sections, _known_names(story))
    memory = memories[int(index) - 1]
    memory.summary = clean_summary
    memory.facts = _lines(facts)
    memory.open_clues = _lines(open_clues)
    _write_memories(config, workspace, memories)
    store.sync_story_files(story_id, len("\n\n".join(sections).strip()), len(sections), len(memories))
    return memory


# ---- AI 퇴고 -----------------------------------------------------------------------------------

def rewrite_plan(instruction: str) -> tuple[list[str], str]:
    """요청을 (지시 목록, 모드) 로. 모드: minimal(오탈자만) · style(문체 전체) · targeted(부분 수정).

    UI 는 빠른 요청과 직접 요청을 ' / ' 로 이어 보낸다. 빠른 요청은 구체적 지시문으로 펼친다.
    """
    parts = [part.strip() for part in str(instruction or "").split(" / ") if part.strip()]
    quick = [part for part in parts if part in REWRITE_DIRECTIVES]
    custom = [part for part in parts if part not in REWRITE_DIRECTIVES]
    if not parts:
        return ["문장을 자연스럽게 다듬는다."], "targeted"
    if quick and not custom and all(MINIMAL_EDIT_RE.search(part) for part in quick):
        mode = "minimal"
    elif custom and all(MINIMAL_EDIT_RE.search(part) for part in custom) and not quick:
        mode = "minimal"
    elif quick:
        mode = "style"
    else:
        mode = "targeted"
    return [REWRITE_DIRECTIVES.get(part, part) for part in parts], mode


def rewrite_temperature(instruction: str) -> float:
    mode = rewrite_plan(instruction)[1]
    return {"minimal": 0.4, "style": 0.8}.get(mode, 0.6)


def _json_payload(text: str) -> dict[str, Any]:
    """JSON 객체 하나를 읽는다. 토큰 상한에 잘렸으면 마지막 완결 항목까지 살린다."""
    cleaned = re.sub(r"```[a-zA-Z]*", "", str(text or "")).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    salvaged = _salvage_json_object(cleaned)
    return salvaged if isinstance(salvaged, dict) else {}


def _parse_plan(raw: str) -> list[dict[str, str]]:
    edits = []
    for item in _json_payload(raw).get("edits", []) or []:
        if not isinstance(item, dict):
            continue
        where = " ".join(str(item.get("where", "")).split())[:80]
        how = " ".join(str(item.get("how", "")).split())[:200]
        if len(where) >= 4 and how:
            edits.append({"where": where, "how": how})
    return edits[:MAX_PLAN_EDITS]


def plan_edits(client: Any, config: AppConfig, story: dict[str, Any], section_text: str, instruction: str) -> list[dict[str, str]]:
    """고쳐 쓰기 전에 '실제로 바꿀 곳' 을 고른다 — 원문 구절과 바꾸는 방법의 목록.

    4B 모델은 고쳐 쓰라고만 하면 원문을 베낀다. 먼저 바꿀 곳을 약속하게 하면 퇴고가 실제로 일어난다.
    최소 수정 모드에서는 계획이 필요 없다. 모델이 JSON 을 못 주면 빈 목록(계획 없이 고쳐 쓴다).
    """
    directives, mode = rewrite_plan(instruction)
    if mode != "style":
        # 최소 수정은 계획이 필요 없고, 부분 수정은 계획 단계가 요청을 잊고 아무 데나 줄인다 (08-23 실측).
        return []
    prompt = "\n".join([
        "[인물 — 이 이름들만 쓴다]", character_sheet(story),
        "[고칠 장의 원문]", section_text.strip(),
        "[수정 요청]", *[f"- {line}" for line in directives],
        f"위 요청대로 이 장을 고치려 한다. 고치거나 더할 곳을 6~{MAX_PLAN_EDITS}개 골라라. 각 항목의 where 는 원문에서 그대로 딴 "
        "짧은 구절(8~20자), how 는 그 자리를 어떻게 바꿀지 또는 그 뒤에 무엇(대사·행동·감각 묘사)을 더할지 40자 이내로 적는다 "
        "(원문 문장을 how 에 되풀이하지 않는다). 문장을 줄이는 것만으로 채우지 않는다. 장의 처음·중간·끝에 고르게 분포시키고, "
        "사건의 순서와 결과는 바꾸지 않는다.",
        '아래 형태의 JSON 객체 하나만 출력한다: {"edits": [{"where": "원문 구절", "how": "바꾸거나 더하는 방법"}]}',
    ])
    edits: list[dict[str, str]] = []
    for attempt, temperature in enumerate((0.7, 0.85)):
        try:
            raw = client.chat(prompt, system=PLAN_SYSTEM_PROMPT, temperature=temperature, max_tokens=1200,
                              json_mode=True, korean_filter=True)
        except Exception as exc:  # noqa: BLE001 - 계획은 보조 단계, 없이도 고쳐 쓴다
            logger.warning("AI rewrite plan failed: %s", exc)
            return edits
        edits = _parse_plan(raw)
        if len(edits) >= 3:
            break
        logger.info("AI rewrite plan attempt %d yielded %d edits (%d chars)", attempt + 1, len(edits), len(raw))
    return edits


SHORTEN_RE = re.compile(r"줄여|짧게|간결|담백|압축|덜어")
LENGTHEN_RE = re.compile(r"늘려|늘리|길게|풍부|더해|보태|확장")


def length_bounds(instruction: str) -> tuple[float, float]:
    """제안 분량이 원문 대비 이 범위를 벗어나면 한 번 더 쓴다 (요청이 줄이기/늘리기면 그쪽은 풀어 준다).

    08-23 실측: 같은 요청도 장에 따라 0.4배로 반토막 나거나 2.6배 반복 루프가 난다. 프롬프트로는 못 잡는다.
    """
    low, high = 0.7, 1.4
    if SHORTEN_RE.search(instruction or ""):
        low = 0.45
    if LENGTHEN_RE.search(instruction or ""):
        high = 1.8
    return low, high


def rewrite_messages(story: dict[str, Any], section_text: str, instruction: str,
                     plan: list[dict[str, str]] | None = None, retry_note: str = "") -> list[dict[str, str]]:
    directives, mode = rewrite_plan(instruction)
    request = "\n".join(f"- {line}" for line in directives)
    plan_block = "\n".join(f"- '{edit['where']}' → {edit['how']}" for edit in (plan or []))
    if mode == "minimal":
        scope = "- 고칠 곳이 아닌 문장은 원문 그대로 둔다."
    elif mode == "style":
        scope = ("- 원문 문장을 그대로 옮겨 적지 않는다. 모든 문단을 요청한 방향으로 새 문장으로 다시 쓴다 "
                 "(사건의 순서와 결과는 유지).")
    else:
        scope = "- 요청에 해당하는 부분을 고치고, 나머지 문장은 원문을 유지한다."
    guide = style_guide(story)
    user = "\n".join(part for part in [
        "[작품 설정]", world_sheet(story),
        "[인물 — 이 이름들만 쓴다]", character_sheet(story),
        "[집필 지침 — 작가가 정한 문체·시점·금기]" if guide else None, guide or None,
        "[수정 요청] 아래 장을 이 요청대로 고쳐 쓴다.", request,
        "[고칠 장의 원문]", section_text.strip(),
        "[수정 요청 — 다시 한번]", request,
        "[수정 계획 — 아래 항목을 모두 반영한다]" if plan_block else None, plan_block or None,
        "[규칙]",
        scope,
        "- 새 사건을 덧붙이지 않는다. 원문 마지막 문단과 같은 상황·같은 장소에서 끝낸다.",
        "- 문단 수는 원문과 비슷하게. 한 줄에 한 문장씩 늘어놓지 않는다.",
        "- 첫 줄은 '### 소제목' 한 줄 (바꾸라는 요청이 없으면 원문 소제목 그대로). 그 뒤는 본문 문단만.",
        "- 새 인물·새 설정을 만들지 않는다. 인물 이름은 위 인물표의 이름만 쓴다.",
        "- 분량은 원문의 8할~12할 (줄이거나 늘리라는 요청이 있으면 그에 따른다).",
        "- 한국어로만 쓴다. 해설·목록·메모 없이 고친 본문만 출력한다.",
        "[이전 시도의 문제 — 이번엔 반드시 피할 것]" if retry_note else None, f"- {retry_note}" if retry_note else None,
    ] if part is not None)
    return [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_note_for(section_text: str, output: str, instruction: str) -> str:
    """제안 분량이 범위를 벗어났을 때 두 번째 시도에 붙일 한 줄. 범위 안이면 빈 문자열."""
    low, high = length_bounds(instruction)
    ratio = len(output) / max(1, len(section_text))
    if ratio < low:
        return (f"분량이 원문의 {ratio * 100:.0f}% 로 너무 짧았다. 원문의 문단과 사건을 하나도 빠뜨리지 말고 "
                "문단마다 원문과 비슷한 길이로 쓴다.")
    if ratio > high:
        return (f"분량이 원문의 {ratio * 100:.0f}% 로 너무 길었다. 같은 말을 되풀이하거나 새 사건을 덧붙이지 말고 "
                "원문과 비슷한 길이로 쓴다.")
    return ""


def stream_rewrite(client: Any, config: AppConfig, story: dict[str, Any], section_text: str, instruction: str,
                   plan: list[dict[str, str]] | None = None, retry_note: str = "") -> Iterator[str]:
    """AI 퇴고를 조각 단위로 낸다 (st.write_stream 에 그대로 넘길 수 있다). 소설은 항상 기본 모델.

    plan 은 plan_edits() 의 결과. 없으면 계획 없이 고쳐 쓴다 (최소 수정 모드나 계획 실패 시).
    """
    llm = config.llm
    options: dict[str, Any] = {}
    if float(getattr(llm, "novel_dry_multiplier", 0.0)) > 0:
        options.update(dry_multiplier=float(llm.novel_dry_multiplier), dry_base=float(llm.novel_dry_base),
                       dry_allowed_length=int(llm.novel_dry_allowed_length))
    max_tokens = int(min(4000, max(800, len(section_text) * 1.2 + 300)))
    logger.info("AI rewrite: %d chars, mode=%s, max_tokens=%d", len(section_text), rewrite_plan(instruction)[1], max_tokens)
    yield from client.stream_messages(
        rewrite_messages(story, section_text, instruction, plan, retry_note),
        temperature=rewrite_temperature(instruction),
        max_tokens=max_tokens,
        adapter=None,
        korean_filter="strict",
        repetition_penalty=float(llm.novel_repetition_penalty),
        **options,
    )
