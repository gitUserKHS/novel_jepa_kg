from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator


class World(BaseModel):
    genre: str
    premise: str
    rules: list[str] = Field(default_factory=list)


class Character(BaseModel):
    name: str
    role: str
    goal: str
    fear: str = ""
    relationship: str = ""

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

    @validator("summary")
    def summary_must_exist(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("scene summary is required")
        return value


class SceneTransition(BaseModel):
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
        return data
