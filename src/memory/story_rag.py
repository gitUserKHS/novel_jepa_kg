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
# The prompt asks for the markers above, but the model decides what it actually
# writes: `<STORY_MEMORY>` and `</STORY_MEMORY>` both occur in practice, and an
# exact-string search silently fails on them, dumping raw JSON into the reader's
# prose. Match the whole family of markers instead of one literal.
MEMORY_TAG_RE = re.compile(
    r"<{1,3}\s*(?P<close>/)?\s*(?P<end>END[_\s]*)?STORY[_\s]*MEMORY\s*>{1,3}",
    re.IGNORECASE,
)
# Longest marker we tolerate, plus slack. The stream filter holds back this much
# of its tail so a marker split across two chunks is still recognised.
MAX_TAG_CHARS = 32
# Field names from the private record. They are JSON keys, so they never appear
# in Korean prose, which makes them a safe signal for a block the model emitted
# without any marker at all.
MEMORY_JSON_KEYS = (
    '"section_index"',
    '"open_clues"',
    '"resolved_clues"',
    '"state_updates"',
    '"state_changes"',
    '"keywords"',
    '"characters"',
    '"summary"',
    '"facts"',
)
_MEMORY_SCAN_CHARS = 600
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[가-힣]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _is_end_tag(match: re.Match[str]) -> bool:
    return bool(match.group("close") or match.group("end"))


def _find_end_tag(text: str, start: int = 0) -> re.Match[str] | None:
    for match in MEMORY_TAG_RE.finditer(text, start):
        if _is_end_tag(match):
            return match
    return None


def _memory_key_hits(text: str) -> int:
    window = text[:_MEMORY_SCAN_CHARS]
    return sum(1 for key in MEMORY_JSON_KEYS if key in window)


def _find_unmarked_memory_start(text: str, *, min_key_hits: int) -> int:
    """Index of a `{` that opens the private record, or -1.

    Used as a backstop for responses where the model dropped the markers but
    still appended the JSON object.
    """
    for match in re.finditer(r"\{", text):
        if _memory_key_hits(text[match.start() :]) >= min_key_hits:
            return match.start()
    return -1


def strip_machine_block(prose: str) -> str:
    """Remove any private continuity record left in reader-facing prose.

    The marker split runs first; this is the guarantee that holds when the model
    ignores the marker contract entirely. Requiring two distinct JSON keys keeps
    ordinary prose containing a brace from being truncated.

    Whitespace at the edges is preserved: the stream filter passes its withheld
    tail through here, and trimming a leading space would glue that tail onto the
    previously emitted word.
    """
    text = MEMORY_TAG_RE.sub("", prose)
    cut = _find_unmarked_memory_start(text, min_key_hits=2)
    if cut >= 0:
        text = text[:cut]
    return re.sub(r"```(?:json)?\s*$", "", text)


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

    def _emit(self, text: str) -> None:
        if text:
            self.callback(text)

    def feed(self, chunk: str) -> None:
        self.buffer += chunk
        while self.buffer:
            if self.hidden:
                end_match = _find_end_tag(self.buffer)
                if end_match is None:
                    self.buffer = self.buffer[-(MAX_TAG_CHARS - 1) :]
                    return
                self.buffer = self.buffer[end_match.end() :]
                self.hidden = False
                continue

            tag = MEMORY_TAG_RE.search(self.buffer)
            if tag is not None:
                self._emit(self.buffer[: tag.start()])
                self.buffer = self.buffer[tag.end() :]
                # A stray closing marker is dropped rather than shown.
                self.hidden = not _is_end_tag(tag)
                continue

            # The record can also arrive with no marker at all. A brace is the
            # only thing that can open one, and it is vanishingly rare in Korean
            # prose, so withhold from it until there is enough text to judge.
            brace = self.buffer.find("{")
            if brace >= 0:
                if _memory_key_hits(self.buffer[brace:]) >= 1:
                    self._emit(self.buffer[:brace])
                    self.buffer = self.buffer[brace:]
                    self.hidden = True
                    return
                if len(self.buffer) - brace < _MEMORY_SCAN_CHARS:
                    self._emit(self.buffer[:brace])
                    self.buffer = self.buffer[brace:]
                    return
                # Enough text followed the brace without a record key: ordinary
                # prose, so release it and look for the next candidate.
                self._emit(self.buffer[: brace + 1])
                self.buffer = self.buffer[brace + 1 :]
                continue

            safe_length = len(self.buffer) - (MAX_TAG_CHARS - 1)
            if safe_length <= 0:
                return
            self._emit(self.buffer[:safe_length])
            self.buffer = self.buffer[safe_length:]

    def finish(self) -> None:
        if self.buffer and not self.hidden:
            # A withheld tail may still hold a truncated record.
            self._emit(strip_machine_block(self.buffer))
        self.buffer = ""


def split_story_memory(
    response: str,
    section_index: int,
    known_names: list[str] | None = None,
) -> tuple[str, StoryMemory]:
    prose = response
    payload = ""
    start = MEMORY_TAG_RE.search(response)
    while start is not None and _is_end_tag(start):
        start = MEMORY_TAG_RE.search(response, start.end())
    if start is not None:
        end_match = _find_end_tag(response, start.end())
        prose = response[: start.start()]
        payload = response[start.end() : end_match.start() if end_match else None]
    else:
        # No marker survived, but the record itself may still be there. Reading
        # it keeps the structured memory instead of silently degrading to the
        # heuristic fallback.
        bare = _find_unmarked_memory_start(response, min_key_hits=2)
        if bare >= 0:
            prose = response[:bare]
            payload = response[bare:]

    prose = strip_machine_block(prose)
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
