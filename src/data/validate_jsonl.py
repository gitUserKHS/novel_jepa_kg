from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.data.scene_schema import SceneTransition


CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def _json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
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
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    errors = []
    for candidate in _json_object_candidates(cleaned):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue

    snippet = cleaned[:500].replace("\n", "\\n")
    detail = errors[-1] if errors else "no balanced JSON object found"
    raise ValueError(f"Could not extract a JSON object: {detail}. Output starts with: {snippet}")


def validate_sample(payload: dict[str, Any]) -> SceneTransition:
    try:
        return SceneTransition(**payload)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc


def validate_jsonl_line(line: str) -> SceneTransition:
    return validate_sample(json.loads(line))
