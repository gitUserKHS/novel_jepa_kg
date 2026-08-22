"""일반 채팅 저장소 — 소비자 DB(SQLite) 안의 별도 테이블.

작품(ConsumerStore)과 독립적으로 계정별 대화 목록과 메시지를 보관한다. 생성은 웹
프로세스에서 모델 서버에 직접 스트리밍하므로 큐/워커를 거치지 않는다.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from src.utils.config import AppConfig
from src.utils.paths import resolve_path

MAX_MESSAGE_CHARS = 20_000
TITLE_CHARS = 40


class ChatStoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.database_path = resolve_path(config, config.consumer.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    adapter TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_owner ON chats(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id, id);
                """
            )

    # ---- chats ----------------------------------------------------------------------------
    def create_chat(self, owner_id: str, *, title: str = "새 대화", adapter: str = "") -> dict[str, Any]:
        chat_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO chats(id, owner_id, title, adapter, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (chat_id, owner_id, title.strip()[:TITLE_CHARS] or "새 대화", adapter, now, now),
            )
        chat = self.get_chat(owner_id, chat_id)
        assert chat is not None
        return chat

    def get_chat(self, owner_id: str, chat_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM chats WHERE id = ? AND owner_id = ?", (chat_id, owner_id)
            ).fetchone()
        return dict(row) if row else None

    def list_chats(self, owner_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = c.id) AS message_count
                FROM chats c WHERE c.owner_id = ? ORDER BY c.updated_at DESC LIMIT ?
                """,
                (owner_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_chat(self, owner_id: str, chat_id: str, title: str) -> None:
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip()[:TITLE_CHARS] or "새 대화", _now(), chat_id),
            )

    def set_adapter(self, owner_id: str, chat_id: str, adapter: str) -> None:
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            connection.execute("UPDATE chats SET adapter = ? WHERE id = ?", (adapter, chat_id))

    def delete_chat(self, owner_id: str, chat_id: str) -> None:
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    # ---- messages -------------------------------------------------------------------------
    def list_messages(self, owner_id: str, chat_id: str) -> list[dict[str, Any]]:
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, role, content, created_at FROM chat_messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_message(self, owner_id: str, chat_id: str, role: str, content: str) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ChatStoreError("지원하지 않는 메시지 역할이야.")
        text = content.strip()
        if not text:
            raise ChatStoreError("빈 메시지는 저장하지 않아.")
        text = text[:MAX_MESSAGE_CHARS]
        chat = self._require(owner_id, chat_id)
        now = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_messages(chat_id, role, content, created_at) VALUES(?, ?, ?, ?)",
                (chat_id, role, text, now),
            )
            title = chat["title"]
            if role == "user" and chat["title"] == "새 대화":
                title = " ".join(text.split())[:TITLE_CHARS]
            connection.execute("UPDATE chats SET updated_at = ?, title = ? WHERE id = ?", (now, title, chat_id))
            message_id = int(cursor.lastrowid)
        return {"id": message_id, "role": role, "content": text, "created_at": now}

    def delete_message_pair(self, owner_id: str, chat_id: str, message_id: int) -> None:
        """사용자 메시지와 그 바로 다음 답변을 함께 지운다."""
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, role FROM chat_messages WHERE id = ? AND chat_id = ?", (message_id, chat_id)
            ).fetchone()
            if row is None:
                return
            ids = [int(row["id"])]
            if row["role"] == "user":
                nxt = connection.execute(
                    "SELECT id, role FROM chat_messages WHERE chat_id = ? AND id > ? ORDER BY id LIMIT 1",
                    (chat_id, message_id),
                ).fetchone()
                if nxt is not None and nxt["role"] == "assistant":
                    ids.append(int(nxt["id"]))
            connection.executemany("DELETE FROM chat_messages WHERE id = ?", [(i,) for i in ids])
            connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))

    def clear_messages(self, owner_id: str, chat_id: str) -> None:
        self._require(owner_id, chat_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            connection.execute("UPDATE chats SET updated_at = ?, title = '새 대화' WHERE id = ?", (_now(), chat_id))

    def _require(self, owner_id: str, chat_id: str) -> dict[str, Any]:
        chat = self.get_chat(owner_id, chat_id)
        if chat is None:
            raise ChatStoreError("이 계정에서 열 수 없는 대화야.")
        return chat
