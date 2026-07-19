from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.generation.consistency import check_name_consistency
from src.memory.story_rag import StoryMemory


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
TRANSITION_CUES = (
    "되었다",
    "되었",
    "바뀌",
    "변했",
    "떠났",
    "도착",
    "잃었",
    "얻었",
    "깨졌",
    "회복",
    "죽었",
    "살아났",
    "때문에",
    "결과",
    "결심",
    "선택",
    "깨달",
    "이동",
    "옮겼",
    "들어갔",
    "나왔",
)


@dataclass(frozen=True)
class StabilityAssessment:
    score: float
    issues: list[str]
    hard_failure: bool


def assess_section_stability(
    section: str,
    memory: StoryMemory,
    *,
    ledger: dict[str, Any],
    characters: str,
    prior_titles: list[str],
    minimum_chars: int,
) -> StabilityAssessment:
    issues: list[str] = []
    hard_failure = False
    normalized = section.strip()
    body = _section_body(normalized)
    title = _section_title(normalized)

    if len(body) < max(200, int(minimum_chars)):
        issues.append(f"section body is too short ({len(body)} < {int(minimum_chars)} chars)")
        hard_failure = True
    if body and body[-1] not in ".!?。！？…\"'”’)]」』":
        issues.append("section appears to end mid-sentence")
        hard_failure = True
    if title and _normalize(title) in {_normalize(item) for item in prior_titles if item.strip()}:
        issues.append(f"subtitle repeats an earlier section: {title}")

    name_check = check_name_consistency(section, characters)
    issues.extend(name_check.issues)
    issues.extend(_state_conflicts(section, memory, ledger))
    issues.extend(_reopened_clues(memory, ledger))
    score = max(0.0, 1.0 - min(1.0, len(issues) * 0.18 + (0.22 if hard_failure else 0.0)))
    return StabilityAssessment(round(score, 4), _unique(issues), hard_failure)


def _state_conflicts(section: str, memory: StoryMemory, ledger: dict[str, Any]) -> list[str]:
    current = {
        (_normalize(str(item.get("entity", ""))), _normalize(str(item.get("attribute", "")))): str(
            item.get("value", "")
        )
        for item in ledger.get("current_states", [])
        if item.get("entity") and item.get("attribute")
    }
    issues: list[str] = []
    for update in memory.state_updates:
        key = (_normalize(update.entity), _normalize(update.attribute))
        previous = current.get(key)
        if not previous or _values_match(previous, update.value):
            continue
        if not _has_explicit_transition(section, memory, update.entity, update.value):
            issues.append(
                f"state changed without an explicit cause: {update.entity}.{update.attribute} "
                f"({previous} -> {update.value})"
            )
    return issues


def _has_explicit_transition(
    section: str,
    memory: StoryMemory,
    entity: str,
    value: str,
) -> bool:
    structured_changes = " ".join(memory.state_changes)
    if (
        _normalize(entity) in _normalize(structured_changes)
        and _normalize(value) in _normalize(structured_changes)
        and any(cue in structured_changes for cue in TRANSITION_CUES)
    ):
        return True

    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", section):
        normalized = _normalize(sentence)
        if _normalize(entity) not in normalized or _normalize(value) not in normalized:
            continue
        if any(cue in sentence for cue in TRANSITION_CUES):
            return True
    return False


def _reopened_clues(memory: StoryMemory, ledger: dict[str, Any]) -> list[str]:
    resolved = [str(item.get("text", "")) for item in ledger.get("resolved_clues", [])]
    issues: list[str] = []
    for open_clue in memory.open_clues:
        for old_clue in resolved:
            if _token_overlap(open_clue, old_clue) >= 0.6:
                issues.append(f"resolved clue was reopened without a new cause: {open_clue}")
                break
    return issues


def _section_title(section: str) -> str:
    first = section.splitlines()[0].strip() if section else ""
    return first[4:].strip() if first.startswith("### ") else ""


def _section_body(section: str) -> str:
    lines = section.splitlines()
    if lines and lines[0].strip().startswith("### "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(_normalize(left)))
    right_tokens = set(TOKEN_RE.findall(_normalize(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _values_match(left: str, right: str) -> bool:
    normalized_left = _normalize(left)
    normalized_right = _normalize(right)
    return normalized_left == normalized_right or _token_overlap(normalized_left, normalized_right) >= 0.75


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return output
