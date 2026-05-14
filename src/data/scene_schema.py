from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, validator


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = _as_text(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        if parts:
            return "; ".join(parts)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " / ".join(text for item in value if (text := _as_text(item)))
    return str(value).strip()


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text for item in value if (text := _as_text(item))]
    if isinstance(value, dict):
        return [f"{key}: {text}" for key, item in value.items() if (text := _as_text(item))]

    text = _as_text(value)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[\n;,，、|]+|\s*/\s*", text) if part.strip()]
    return parts or [text]


class World(BaseModel):
    genre: str
    premise: str
    rules: list[str] = Field(default_factory=list)

    @validator("genre", "premise", pre=True)
    def normalize_text(cls, value: Any) -> str:
        return _as_text(value)

    @validator("rules", pre=True)
    def normalize_rules(cls, value: Any) -> list[str]:
        return _as_text_list(value)


class Character(BaseModel):
    name: str
    role: str
    goal: str
    fear: str = ""
    relationship: str = ""

    @validator("name", "role", "goal", "fear", "relationship", pre=True)
    def normalize_text(cls, value: Any) -> str:
        return _as_text(value)

    @validator("goal")
    def goal_must_exist(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("character goal is required")
        return value


class Scene(BaseModel):
    summary: str
    emotion: str
    conflict: str
    state: list[str] = Field(default_factory=list)
    plot_function: str

    @validator("summary", "emotion", "conflict", "plot_function", pre=True)
    def normalize_text(cls, value: Any) -> str:
        return _as_text(value)

    @validator("state", pre=True)
    def normalize_state(cls, value: Any) -> list[str]:
        return _as_text_list(value)

    @validator("summary")
    def summary_must_exist(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("scene summary is required")
        return value


class SceneTransition(BaseModel):
    id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    world: World
    characters: list[Character]
    scene_t: Scene
    scene_t_plus_1: Scene

    def to_jsonable(self, sample_id: int | None = None) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            data = self.model_dump()
        else:
            data = self.dict()
        if sample_id is not None:
            data["id"] = sample_id
        if data.get("id") is None:
            data.pop("id", None)
        return data
