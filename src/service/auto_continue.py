"""자동 이어쓰기 — 사람이 다음 전개를 적지 않아도 AI 가 정해서 이어 쓴다.

작품의 `auto_continue` 가 켜져 있으면 워커가 턴을 마칠 때마다 다음 턴을 큐에 넣는다(`origin='auto'`).
자동 턴의 지시는 워커가 작업을 집는 순간 이야기 지도와 요약 메모리를 보고 한두 문장으로 정하고, 그 문장을
작업 기록에 남긴다 — 무엇을 시켰는지 말풍선에서 읽을 수 있어야 사람이 언제든 끼어들 수 있다.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.memory.story_outline import load_story_outline, outline_context
from src.memory.story_rag import format_hierarchical_story_context, load_story_memories
from src.service.story_sheets import character_sheet, world_sheet
from src.service.story_workspace import StoryWorkspace, read_draft, split_sections
from src.utils.config import AppConfig

logger = logging.getLogger(__name__)

AUTO_PLACEHOLDER = "🤖 AI 가 다음 전개를 정해 이어 써."
AUTO_FALLBACK_DIRECTION = "이야기 지도의 다음 단계로 전개를 한 걸음 옮긴다. 미해결 단서 하나를 건드리고 인물의 목표를 시험한다."
DIRECTION_SYSTEM_PROMPT = (
    "당신은 한국어 장편 소설의 기획 편집자다. 다음 장에서 일어날 사건을 작가에게 지시하듯 구체적으로 정한다. "
    "지시 문장만 말한다."
)
MAX_DIRECTION_CHARS = 200


def propose_next_direction(client: Any, config: AppConfig, story: dict[str, Any], workspace: StoryWorkspace) -> str:
    """다음 턴의 전개 지시 한두 문장. 모델이 없거나 실패하면 빈 문자열 (호출자가 AUTO_FALLBACK_DIRECTION 을 쓴다)."""
    sections = split_sections(read_draft(workspace.draft))
    memories = load_story_memories(workspace.memory)
    outline = load_story_outline(workspace.outline)
    overall_target = max(1000, int(story.get("target_chars") or config.generation.target_novel_chars))
    section_target = max(600, int(config.generation.section_min_chars))
    planned = max(int(config.generation.section_count), math.ceil(overall_target / section_target), 1)
    next_index = len(sections) + 1
    if outline is not None and outline.beats:
        beat_text, _beat = outline_context(outline, section_index=next_index, planned_sections=planned)
    else:
        beat_text = "(이야기 지도 없음 — 미해결 갈등을 한 단계 전진시킨다)"
    if memories:
        context, _ledger = format_hierarchical_story_context(
            memories, [], "", max(400, int(config.generation.story_memory_context_chars)),
            group_size=config.generation.story_summary_group_size,
        )
        last = memories[-1]
        open_clues = "; ".join(str(item) for item in last.open_clues[:4]) or "(없음)"
    else:
        context, open_clues = "(아직 쓴 장이 없다 — 첫 장면이다)", "(없음)"
    remaining = max(0, overall_target - len("\n\n".join(sections)))
    prompt = "\n".join([
        "[작품 설정]", world_sheet(story),
        "[인물 — 이 이름들만 쓴다]", character_sheet(story),
        f"[이야기 지도 — {next_index}장이 맡을 단계]", beat_text.strip(),
        "[지금까지의 줄거리와 현재 상태]", context.strip(),
        f"[직전 장의 미해결 단서] {open_clues}",
        f"[남은 분량] 약 {remaining:,}자 (0에 가까우면 결말로 수렴한다)",
        f"{next_index}장에서 일어날 사건을 한두 문장으로 정한다. 작가가 편집자에게 지시하듯 '누가 무엇을 하게 한다' 로 "
        "구체적으로 쓴다 (예: 박 노인이 20년 전 일을 털어놓게 해줘). 지나간 사건을 되풀이하지 않고 이야기 지도의 "
        "단계를 한 걸음 옮긴다. 새 인물·새 설정을 만들지 않는다. 한국어로 지시 문장만 출력한다 (제목·해설·따옴표 없이).",
    ])
    try:
        raw = client.chat(prompt, system=DIRECTION_SYSTEM_PROMPT, temperature=0.7, max_tokens=160, korean_filter=True)
    except Exception as exc:  # noqa: BLE001 - 전개 제안은 보조 단계, 실패하면 기본 지시로 이어 쓴다
        logger.warning("Auto direction failed: %s", exc)
        return ""
    return clean_direction(raw)


def clean_direction(raw: str) -> str:
    """모델 출력에서 지시 문장만: 펜스·따옴표·라벨을 벗기고 앞 두 문장, 200자 이내."""
    quotes = " \"'“”‘’「」『』"
    text = re.sub(r"```[a-zA-Z]*", "", str(raw or "")).strip(quotes)
    text = re.sub(r"^\s*(지시|다음 전개|전개|제안)\s*[:：]\s*", "", text.strip(), flags=re.MULTILINE)
    text = " ".join(text.split()).strip(quotes)
    if not text:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    picked = " ".join(sentences[:2]) if sentences else text
    picked = picked.strip()[:MAX_DIRECTION_CHARS].strip()
    return picked if len(picked) >= 6 else ""
