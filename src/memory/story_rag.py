from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from src.utils.paths import ensure_parent


MEMORY_START = "<<<STORY_MEMORY>>>"
MEMORY_END = "<<<END_STORY_MEMORY>>>"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[가-힣]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class StateUpdate(BaseModel):
    entity: str
    attribute: str
    value: str


class KnowledgeTriple(BaseModel):
    source: str
    relation: str
    target: str


class StoryMemory(BaseModel):
    section_index: int
    title: str = ""
    summary: str = ""
    characters: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    open_clues: list[str] = Field(default_factory=list)
    resolved_clues: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    state_changes: list[str] = Field(default_factory=list)
    state_updates: list[StateUpdate] = Field(default_factory=list)
    relations: list[KnowledgeTriple] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


def story_memory_instruction(section_index: int) -> str:
    return f"""

[Private story-memory record]
After the prose, append one compact machine-readable object between these exact markers.
This block is private continuity data and is removed before the novel is shown.
Do not use markdown fences. Keep each list short and only record facts established in this section.

{MEMORY_START}
{{
  "section_index": {section_index},
  "title": "section subtitle",
  "summary": "one concise sentence",
  "characters": ["characters who actually appeared"],
  "facts": ["newly established fact"],
  "open_clues": ["unresolved clue or promise"],
  "resolved_clues": ["clue resolved here"],
  "locations": ["important location"],
  "state_changes": ["character, relationship, object, or goal state change"],
  "state_updates": [
    {{"entity": "character or object", "attribute": "location|emotion|goal|status|owner", "value": "latest value"}}
  ],
  "relations": [
    {{"source": "entity", "relation": "possesses|trusts|hides|seeks|located_at|causes", "target": "entity or fact"}}
  ],
  "keywords": ["3 to 8 retrieval keywords"]
}}
{MEMORY_END}
""".rstrip()


class StoryMemoryStreamFilter:
    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self.buffer = ""
        self.hidden = False

    def feed(self, chunk: str) -> None:
        self.buffer += chunk
        while self.buffer:
            if self.hidden:
                end_index = self.buffer.find(MEMORY_END)
                if end_index < 0:
                    self.buffer = self.buffer[-(len(MEMORY_END) - 1) :]
                    return
                self.buffer = self.buffer[end_index + len(MEMORY_END) :]
                self.hidden = False
                continue

            start_index = self.buffer.find(MEMORY_START)
            if start_index >= 0:
                visible = self.buffer[:start_index]
                if visible:
                    self.callback(visible)
                self.buffer = self.buffer[start_index + len(MEMORY_START) :]
                self.hidden = True
                continue

            safe_length = len(self.buffer) - (len(MEMORY_START) - 1)
            if safe_length <= 0:
                return
            self.callback(self.buffer[:safe_length])
            self.buffer = self.buffer[safe_length:]

    def finish(self) -> None:
        if self.buffer and not self.hidden:
            self.callback(self.buffer)
        self.buffer = ""


def split_story_memory(
    response: str,
    section_index: int,
    known_names: list[str] | None = None,
) -> tuple[str, StoryMemory]:
    prose = response
    payload = ""
    start_index = response.find(MEMORY_START)
    if start_index >= 0:
        end_index = response.find(MEMORY_END, start_index + len(MEMORY_START))
        prose = response[:start_index]
        payload = response[start_index + len(MEMORY_START) : end_index if end_index >= 0 else None]

    title = _section_title(prose)
    memory = _parse_memory_payload(payload, section_index, title)
    if memory is None:
        memory = _fallback_memory(prose, section_index, title, known_names or [])
    else:
        memory.section_index = section_index
        memory.title = memory.title.strip() or title
        memory.characters = _unique(memory.characters)
        memory.facts = _unique(memory.facts)
        memory.open_clues = _unique(memory.open_clues)
        memory.resolved_clues = _unique(memory.resolved_clues)
        memory.locations = _unique(memory.locations)
        memory.state_changes = _unique(memory.state_changes)
        memory.state_updates = _unique_models(memory.state_updates)
        memory.relations = _unique_models(memory.relations)
        memory.keywords = _unique(memory.keywords)
    return prose.strip(), memory


def retrieve_story_memories(
    memories: list[StoryMemory],
    query: str,
    top_k: int,
    current_section: int,
) -> list[tuple[StoryMemory, float]]:
    if not memories or top_k <= 0:
        return []

    query_tokens = _tokens(query)
    documents = [_tokens(_memory_text(memory)) for memory in memories]
    document_frequency = Counter(token for tokens in documents for token in set(tokens))
    document_count = len(documents)
    query_vector = _tfidf_vector(query_tokens, document_frequency, document_count)
    scored: list[tuple[StoryMemory, float]] = []

    for memory, tokens in zip(memories, documents, strict=True):
        memory_vector = _tfidf_vector(tokens, document_frequency, document_count)
        semantic = _cosine(query_vector, memory_vector)
        shared_names = len(set(query_tokens) & set(_tokens(" ".join(memory.characters))))
        recency = 1.0 / max(1, current_section - memory.section_index)
        unresolved_bonus = 0.08 if memory.open_clues else 0.0
        score = semantic + min(0.24, shared_names * 0.08) + recency * 0.12 + unresolved_bonus
        scored.append((memory, score))

    scored.sort(key=lambda item: (item[1], item[0].section_index), reverse=True)
    return scored[: min(top_k, len(scored))]


def format_story_memory_context(
    retrieved: list[tuple[StoryMemory, float]],
    max_chars: int,
) -> str:
    if not retrieved:
        return "(no prior story memories yet)"

    blocks: list[str] = []
    for memory, _score in retrieved:
        lines = [
            f"[Section {memory.section_index}: {memory.title}]",
            f"- Summary: {memory.summary}",
        ]
        _append_list(lines, "Characters", memory.characters)
        _append_list(lines, "Established facts", memory.facts)
        _append_list(lines, "Open clues/promises", memory.open_clues)
        _append_list(lines, "Resolved clues", memory.resolved_clues)
        _append_list(lines, "Locations", memory.locations)
        _append_list(lines, "State changes", memory.state_changes)
        block = "\n".join(lines)
        candidate = "\n\n".join([*blocks, block])
        if blocks and len(candidate) > max_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks)[:max_chars]


def format_hierarchical_story_context(
    memories: list[StoryMemory],
    retrieved: list[tuple[StoryMemory, float]],
    query: str,
    max_chars: int,
    group_size: int = 4,
) -> tuple[str, dict]:
    ledger = build_story_ledger(memories, group_size=group_size)
    if not memories:
        return "(no prior story memories yet)", ledger

    query_tokens = set(_tokens(query))
    blocks: list[str] = []

    states = sorted(
        ledger["current_states"],
        key=lambda item: (
            _token_overlap(query_tokens, _tokens(f"{item['entity']} {item['attribute']} {item['value']}")),
            item["section_index"],
        ),
        reverse=True,
    )
    if states:
        state_lines = [
            f"- {item['entity']} / {item['attribute']} = {item['value']} (section {item['section_index']})"
            for item in states[:14]
        ]
        blocks.append("[Current state ledger]\n" + "\n".join(state_lines))

    if ledger["open_clues"]:
        clue_lines = [
            f"- {item['text']} (opened in section {item['section_index']})"
            for item in ledger["open_clues"][:10]
        ]
        blocks.append("[Open clues and promises]\n" + "\n".join(clue_lines))

    relations = sorted(
        ledger["relations"],
        key=lambda item: (
            _token_overlap(query_tokens, _tokens(f"{item['source']} {item['relation']} {item['target']}")),
            item["section_index"],
        ),
        reverse=True,
    )
    if relations:
        relation_lines = [
            f"- {item['source']} --{item['relation']}--> {item['target']} (section {item['section_index']})"
            for item in relations[:14]
        ]
        blocks.append("[Relevant knowledge graph]\n" + "\n".join(relation_lines))

    summaries = ledger["hierarchical_summaries"]
    if summaries:
        summary_lines = [
            f"- Sections {item['start_section']}-{item['end_section']}: {item['summary']}"
            for item in summaries[-4:]
        ]
        blocks.append("[Compressed story timeline]\n" + "\n".join(summary_lines))

    memory_context = format_story_memory_context(retrieved, max(600, max_chars // 2))
    if retrieved:
        blocks.append("[Retrieved section memories]\n" + memory_context)

    selected: list[str] = []
    for block in blocks:
        candidate = "\n\n".join([*selected, block])
        if selected and len(candidate) > max_chars:
            remaining = max_chars - len("\n\n".join(selected)) - 2
            if remaining >= 200:
                selected.append(block[:remaining])
            break
        selected.append(block)
    return "\n\n".join(selected)[:max_chars], ledger


def write_story_memories(path: Path, memories: list[StoryMemory]) -> str:
    ensure_parent(path)
    content = "".join(
        json.dumps(memory.model_dump(), ensure_ascii=False) + "\n"
        for memory in memories
    )
    path.write_text(content, encoding="utf-8")
    return str(path)


def load_story_memories(path: Path) -> list[StoryMemory]:
    if not path.exists():
        return []
    memories: list[StoryMemory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            memories.append(StoryMemory.model_validate(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return memories


def build_story_ledger(memories: list[StoryMemory], group_size: int = 4) -> dict:
    state_map: dict[tuple[str, str], dict] = {}
    relation_map: dict[tuple[str, str, str], dict] = {}
    open_clues: list[dict] = []
    resolved_clues: list[dict] = []

    for memory in sorted(memories, key=lambda item: item.section_index):
        for update in memory.state_updates:
            key = (_normalize_key(update.entity), _normalize_key(update.attribute))
            if all(key):
                state_map[key] = {
                    **update.model_dump(),
                    "section_index": memory.section_index,
                }
        for change in memory.state_changes:
            state_map[(_normalize_key(change), "narrative_state")] = {
                "entity": change,
                "attribute": "narrative_state",
                "value": change,
                "section_index": memory.section_index,
            }
        for relation in memory.relations:
            key = (
                _normalize_key(relation.source),
                _normalize_key(relation.relation),
                _normalize_key(relation.target),
            )
            if all(key):
                relation_map[key] = {
                    **relation.model_dump(),
                    "section_index": memory.section_index,
                }
        for clue in memory.open_clues:
            if not any(_clue_matches(clue, item["text"]) for item in resolved_clues):
                open_clues.append({"text": clue, "section_index": memory.section_index})
        for clue in memory.resolved_clues:
            resolved = {"text": clue, "section_index": memory.section_index}
            resolved_clues.append(resolved)
            open_clues = [item for item in open_clues if not _clue_matches(item["text"], clue)]

    chunk_size = max(2, int(group_size))
    hierarchical_summaries: list[dict] = []
    for start in range(0, len(memories), chunk_size):
        group = memories[start : start + chunk_size]
        summaries = [memory.summary.strip() for memory in group if memory.summary.strip()]
        if not group or not summaries:
            continue
        hierarchical_summaries.append(
            {
                "start_section": group[0].section_index,
                "end_section": group[-1].section_index,
                "summary": " ".join(summaries)[:1200],
            }
        )

    return {
        "version": 1,
        "section_count": len(memories),
        "current_states": sorted(state_map.values(), key=lambda item: item["section_index"], reverse=True),
        "relations": sorted(relation_map.values(), key=lambda item: item["section_index"], reverse=True),
        "open_clues": sorted(open_clues, key=lambda item: item["section_index"], reverse=True),
        "resolved_clues": sorted(resolved_clues, key=lambda item: item["section_index"], reverse=True),
        "hierarchical_summaries": hierarchical_summaries,
    }


def write_story_ledger(path: Path, memories: list[StoryMemory], group_size: int = 4) -> str:
    ensure_parent(path)
    ledger = build_story_ledger(memories, group_size=group_size)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _parse_memory_payload(
    payload: str,
    section_index: int,
    title: str,
) -> StoryMemory | None:
    cleaned = payload.strip()
    if not cleaned:
        return None
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start < 0 or object_end <= object_start:
        return None
    try:
        raw = json.loads(cleaned[object_start : object_end + 1])
        raw.setdefault("section_index", section_index)
        raw.setdefault("title", title)
        return StoryMemory.model_validate(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _fallback_memory(
    prose: str,
    section_index: int,
    title: str,
    known_names: list[str],
) -> StoryMemory:
    cleaned = re.sub(r"^###\s+.+$", "", prose, count=1, flags=re.MULTILINE).strip()
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(cleaned) if sentence.strip()]
    summary_parts = sentences[:1] + (sentences[-1:] if len(sentences) > 1 else [])
    summary = " ".join(summary_parts)[:500] or cleaned[:500]
    characters = [name for name in known_names if name and name in prose]
    keywords = [
        token
        for token, _count in Counter(_tokens(f"{title} {summary}")).most_common(8)
        if token not in {"그리고", "그러나", "하지만", "그것", "그녀", "그는"}
    ]
    return StoryMemory(
        section_index=section_index,
        title=title,
        summary=summary,
        characters=characters,
        facts=summary_parts[:2],
        keywords=keywords,
    )


def _section_title(prose: str) -> str:
    for line in prose.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            return stripped[4:].strip()[:80]
    return ""


def _memory_text(memory: StoryMemory) -> str:
    values = [
        memory.title,
        memory.summary,
        *memory.characters,
        *memory.facts,
        *memory.open_clues,
        *memory.resolved_clues,
        *memory.locations,
        *memory.state_changes,
        *(f"{item.entity} {item.attribute} {item.value}" for item in memory.state_updates),
        *(f"{item.source} {item.relation} {item.target}" for item in memory.relations),
        *memory.keywords,
    ]
    return " ".join(value for value in values if value)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _tfidf_vector(
    tokens: list[str],
    document_frequency: Counter[str],
    document_count: int,
) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(1, sum(counts.values()))
    return {
        token: (count / total) * (math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0)
        for token, count in counts.items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _append_list(lines: list[str], label: str, values: list[str]) -> None:
    if values:
        lines.append(f"- {label}: {'; '.join(values[:5])}")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _unique_models(values: list[BaseModel]) -> list:
    result: list[BaseModel] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value.model_dump(), ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _token_overlap(left: set[str], right: list[str]) -> float:
    right_set = set(right)
    if not left or not right_set:
        return 0.0
    return len(left & right_set) / max(1, len(right_set))


def _clue_matches(left: str, right: str) -> bool:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return _normalize_key(left) == _normalize_key(right)
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.5
