from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.data.scene_schema import SceneTransition


CODE_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def validate_sample(payload: dict[str, Any]) -> SceneTransition:
    try:
        return SceneTransition(**payload)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc


def validate_jsonl_line(line: str) -> SceneTransition:
    return validate_sample(json.loads(line))
