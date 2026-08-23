"""작품 템플릿 저장소 — 사용자가 직접 적은 작품 설정(장르·소재·세계관·인물·집필 지침)을 이름 붙여 보관.

소비자 DB(SQLite) 안의 별도 테이블. 같은 이름으로 저장하면 덮어쓴다 (계정별로 이름이 유일).
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from src.service.consumer_store import STYLE_GUIDE_CHARS
from src.utils.config import AppConfig
from src.utils.paths import resolve_path

TEMPLATE_NAME_CHARS = 60
TEMPLATE_FIELDS = ("title", "genre", "premise", "world", "protagonist", "characters", "style_guide")
# 작품 설정 양식·설정 수정 폼·템플릿이 모두 이 한도를 쓴다 (consumer_app 이 import).
FIELD_LIMITS = {
    "title": 100, "genre": 80, "premise": 1500, "world": 4000,
    "protagonist": 2000, "characters": 3000, "style_guide": STYLE_GUIDE_CHARS,
}


class TemplateStoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class StoryTemplateStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.database_path = resolve_path(config, config.consumer.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS story_templates (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    genre TEXT NOT NULL DEFAULT '',
                    premise TEXT NOT NULL DEFAULT '',
                    world TEXT NOT NULL DEFAULT '',
                    protagonist TEXT NOT NULL DEFAULT '',
                    characters TEXT NOT NULL DEFAULT '',
                    style_guide TEXT NOT NULL DEFAULT '',
                    target_chars INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_story_templates_owner ON story_templates(owner_id, updated_at);
                """
            )

    # ---- CRUD -----------------------------------------------------------------------------
    def save_template(self, owner_id: str, *, name: str, target_chars: int = 0, **fields: str) -> dict[str, Any]:
        """이름으로 저장. 같은 이름이 있으면 내용을 덮어쓴다."""
        clean_name = " ".join(str(name or "").split())[:TEMPLATE_NAME_CHARS]
        if not clean_name:
            raise TemplateStoreError("템플릿 이름을 적어줘.")
        unknown = set(fields) - set(TEMPLATE_FIELDS)
        if unknown:
            raise TemplateStoreError(f"지원하지 않는 템플릿 항목: {', '.join(sorted(unknown))}")
        values = {key: str(fields.get(key, "") or "").replace("\r", "").strip()[: FIELD_LIMITS[key]] for key in TEMPLATE_FIELDS}
        if not any(values[key] for key in ("genre", "premise", "world", "protagonist", "characters", "style_guide")):
            raise TemplateStoreError("비어 있는 템플릿은 저장하지 않아. 장르나 세계관을 하나라도 적어줘.")
        now = _now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM story_templates WHERE owner_id = ? AND name = ?", (owner_id, clean_name)
            ).fetchone()
            if existing is None:
                template_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO story_templates(id, owner_id, name, title, genre, premise, world, protagonist,
                        characters, style_guide, target_chars, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (template_id, owner_id, clean_name, values["title"], values["genre"], values["premise"],
                     values["world"], values["protagonist"], values["characters"], values["style_guide"],
                     int(target_chars or 0), now, now),
                )
            else:
                template_id = str(existing["id"])
                connection.execute(
                    """
                    UPDATE story_templates SET title = ?, genre = ?, premise = ?, world = ?, protagonist = ?,
                        characters = ?, style_guide = ?, target_chars = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (values["title"], values["genre"], values["premise"], values["world"], values["protagonist"],
                     values["characters"], values["style_guide"], int(target_chars or 0), now, template_id),
                )
        template = self.get_template(owner_id, template_id)
        assert template is not None
        return template

    def list_templates(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM story_templates WHERE owner_id = ? ORDER BY updated_at DESC, rowid DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_template(self, owner_id: str, template_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM story_templates WHERE id = ? AND owner_id = ?", (template_id, owner_id)
            ).fetchone()
        return dict(row) if row else None

    def delete_template(self, owner_id: str, template_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM story_templates WHERE id = ? AND owner_id = ?", (template_id, owner_id)
            )
        if not cursor.rowcount:
            raise TemplateStoreError("이 계정에서 지울 수 없는 템플릿이야.")


def template_from_story(story: dict[str, Any]) -> dict[str, Any]:
    """작품 설정을 save_template 에 넘길 수 있는 항목으로 추린다."""
    return {key: str(story.get(key, "") or "") for key in TEMPLATE_FIELDS}
