from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.service.security import (
    build_session_token,
    generate_secret,
    hash_password,
    hash_session_secret,
    hash_story_secret,
    split_session_token,
    verify_password,
    verify_session_secret,
)
from src.service.story_workspace import StoryWorkspace
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_FAILED_RECOVERABLE = "failed_recoverable"
OUTSTANDING_JOB_STATUSES = (JOB_QUEUED, JOB_RUNNING)
CREATIVITY_LEVELS = {"stable": 0.20, "balanced": 0.35, "bold": 0.50}
MAINTENANCE_OFF = "0"
MAINTENANCE_ACTIVE = "1"
MAINTENANCE_DRAINING = "draining"


class ConsumerStoreError(RuntimeError):
    pass


class MaintenanceModeError(ConsumerStoreError):
    pass


class StoryBusyError(ConsumerStoreError):
    pass


class AccountExistsError(ConsumerStoreError):
    pass


class AuthorizationError(ConsumerStoreError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_time(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec="seconds")


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    return {
        "id": value["id"],
        "username": value["username"],
        "display_name": value["display_name"],
        "created_at": value["created_at"],
        "updated_at": value["updated_at"],
    }


def _public_story(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("key_salt", None)
    value.pop("key_hash", None)
    value["research_consent"] = bool(value["research_consent"])
    return value


def _normalize_username(username: str) -> str:
    return username.strip().casefold()


class ConsumerStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.database_path = resolve_path(config, config.consumer.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        resolve_path(config, config.consumer.story_root).mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_norm TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_salt TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS user_sessions_expiry
                    ON user_sessions(expires_at);

                CREATE TABLE IF NOT EXISTS stories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    key_salt TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    premise TEXT NOT NULL,
                    world TEXT NOT NULL,
                    protagonist TEXT NOT NULL,
                    characters TEXT NOT NULL DEFAULT '',
                    target_chars INTEGER NOT NULL,
                    current_chars INTEGER NOT NULL DEFAULT 0,
                    section_count INTEGER NOT NULL DEFAULT 0,
                    research_consent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    completed_at TEXT,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                    instruction TEXT NOT NULL,
                    creativity_profile TEXT NOT NULL,
                    requested_chars INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    model_version TEXT,
                    error_public TEXT,
                    start_section_count INTEGER NOT NULL DEFAULT 0,
                    result_section_count INTEGER NOT NULL DEFAULT 0,
                    result_chars INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    finished_at TEXT,
                    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'failed_recoverable')),
                    CHECK (creativity_profile IN ('stable', 'balanced', 'bold'))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_outstanding_job_per_story
                    ON jobs(story_id)
                    WHERE status IN ('queued', 'running');
                CREATE UNIQUE INDEX IF NOT EXISTS one_running_job_globally
                    ON jobs((1))
                    WHERE status = 'running';
                CREATE INDEX IF NOT EXISTS jobs_fifo ON jobs(status, created_at, id);

                CREATE TABLE IF NOT EXISTS section_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    section_index INTEGER NOT NULL,
                    model_version TEXT,
                    creativity_profile TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(story_id, section_index)
                );

                CREATE TABLE IF NOT EXISTS service_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            story_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(stories)").fetchall()
            }
            if "owner_id" not in story_columns:
                connection.execute("ALTER TABLE stories ADD COLUMN owner_id TEXT")
            if "completed_at" not in story_columns:
                connection.execute("ALTER TABLE stories ADD COLUMN completed_at TEXT")
                # Legacy stories finished under the old char-count rule stay closed.
                connection.execute(
                    """
                    UPDATE stories SET completed_at = updated_at
                    WHERE completed_at IS NULL AND current_chars >= target_chars
                    """
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS stories_owner_updated ON stories(owner_id, updated_at DESC)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO service_state(key, value, updated_at) VALUES('maintenance', '0', ?)",
                (iso_time(),),
            )

    def create_user(self, *, username: str, display_name: str, password: str) -> dict[str, Any]:
        clean_username = username.strip()
        normalized = _normalize_username(clean_username)
        clean_display_name = display_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,39}", clean_username):
            raise ValueError("아이디는 영문 또는 숫자로 시작하는 3~40자의 영문, 숫자, ., _, -만 사용할 수 있어.")
        if not clean_display_name or len(clean_display_name) > 40:
            raise ValueError("표시 이름은 1~40자로 입력해줘.")
        if len(password) < 8 or len(password) > 128:
            raise ValueError("비밀번호는 8~128자로 입력해줘.")

        user_id = uuid.uuid4().hex
        salt, digest = hash_password(password)
        now = iso_time()
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users(
                        id, username, username_norm, display_name, password_salt,
                        password_hash, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        clean_username,
                        normalized,
                        clean_display_name,
                        salt,
                        digest,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise AccountExistsError("이미 사용 중인 아이디야.") from exc
        user = self.get_user(user_id)
        assert user is not None
        return user

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ? AND disabled = 0",
                (user_id,),
            ).fetchone()
        return _public_user(row) if row else None

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        normalized = _normalize_username(username)
        if not normalized or not password:
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username_norm = ? AND disabled = 0",
                (normalized,),
            ).fetchone()
        if row is None or not verify_password(password, row["password_salt"], row["password_hash"]):
            return None
        return _public_user(row)

    def create_user_session(self, user_id: str) -> str:
        if self.get_user(user_id) is None:
            raise AuthorizationError("로그인할 수 없는 계정이야.")
        session_id = uuid.uuid4().hex
        secret = generate_secret()
        salt, digest = hash_session_secret(secret)
        now = utc_now()
        expires = now + timedelta(days=self.config.consumer.auth_session_days)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO user_sessions(
                    id, user_id, token_salt, token_hash, created_at, last_seen_at, expires_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    salt,
                    digest,
                    iso_time(now),
                    iso_time(now),
                    iso_time(expires),
                ),
            )
        return build_session_token(session_id, secret)

    def authenticate_user_session(self, token: str) -> dict[str, Any] | None:
        try:
            session_id, secret = split_session_token(token)
        except ValueError:
            return None
        now = iso_time()
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.*, user_sessions.token_salt, user_sessions.token_hash,
                       user_sessions.expires_at
                FROM user_sessions
                JOIN users ON users.id = user_sessions.user_id
                WHERE user_sessions.id = ? AND users.disabled = 0
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            if str(row["expires_at"]) <= now:
                connection.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
                return None
            if not verify_session_secret(secret, row["token_salt"], row["token_hash"]):
                return None
            connection.execute(
                "UPDATE user_sessions SET last_seen_at = ? WHERE id = ?",
                (now, session_id),
            )
        return _public_user(row)

    def revoke_user_session(self, token: str) -> None:
        try:
            session_id, secret = split_session_token(token)
        except ValueError:
            return
        with self.connect() as connection:
            row = connection.execute(
                "SELECT token_salt, token_hash FROM user_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row and verify_session_secret(secret, row["token_salt"], row["token_hash"]):
                connection.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))

    def purge_expired_user_sessions(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM user_sessions WHERE expires_at <= ?",
                (iso_time(),),
            )
        return int(cursor.rowcount)

    def create_story(
        self,
        owner_id: str,
        *,
        title: str,
        genre: str,
        premise: str,
        world: str,
        protagonist: str,
        characters: str = "",
        target_chars: int | None = None,
        research_consent: bool = False,
    ) -> dict[str, Any]:
        if self.get_user(owner_id) is None:
            raise AuthorizationError("로그인이 필요한 작업이야.")
        fields = {
            "title": title.strip(),
            "genre": genre.strip(),
            "premise": premise.strip(),
            "world": world.strip(),
            "protagonist": protagonist.strip(),
            "characters": characters.strip(),
        }
        if any(not fields[name] for name in ("title", "genre", "premise", "world", "protagonist")):
            raise ValueError("제목, 장르, 소재, 세계관, 주인공을 모두 입력해줘.")
        actual_target = int(target_chars or self.config.consumer.default_target_chars)
        if actual_target < self.config.consumer.min_target_chars:
            raise ValueError(
                f"전체 목표 분량은 {self.config.consumer.min_target_chars:,}자 이상이어야 해."
            )
        if actual_target > self.config.consumer.max_target_chars:
            raise ValueError(f"전체 목표 분량은 {self.config.consumer.max_target_chars:,}자 이하여야 해.")

        story_id = uuid.uuid4().hex
        secret = generate_secret()
        salt, digest = hash_story_secret(secret)
        now = utc_now()
        expires = now + timedelta(days=self.config.consumer.retention_days)
        workspace = StoryWorkspace.for_story(self.config, story_id, create=True)
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO stories(
                        id, owner_id, key_salt, key_hash, title, genre, premise, world, protagonist,
                        characters, target_chars, research_consent, created_at, updated_at, expires_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        story_id,
                        owner_id,
                        salt,
                        digest,
                        fields["title"],
                        fields["genre"],
                        fields["premise"],
                        fields["world"],
                        fields["protagonist"],
                        fields["characters"],
                        actual_target,
                        int(research_consent),
                        iso_time(now),
                        iso_time(now),
                        iso_time(expires),
                    ),
                )
        except Exception:
            workspace.delete()
            raise
        story = self.get_owned_story(owner_id, story_id)
        assert story is not None
        return story

    def get_story(self, story_id: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        where = "id = ?" if include_deleted else "id = ? AND deleted_at IS NULL"
        with self.connect() as connection:
            row = connection.execute(f"SELECT * FROM stories WHERE {where}", (story_id,)).fetchone()
        return _public_story(row) if row else None

    def get_owned_story(self, owner_id: str, story_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM stories
                WHERE id = ? AND owner_id = ? AND deleted_at IS NULL AND expires_at > ?
                """,
                (story_id, owner_id, iso_time()),
            ).fetchone()
        return _public_story(row) if row else None

    def list_owned_stories(self, owner_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM stories
                WHERE owner_id = ? AND deleted_at IS NULL AND expires_at > ?
                ORDER BY updated_at DESC
                """,
                (owner_id, iso_time()),
            ).fetchall()
        return [_public_story(row) for row in rows]

    def enqueue_job(
        self,
        owner_id: str,
        story_id: str,
        *,
        instruction: str,
        creativity_profile: str,
        requested_chars: int,
    ) -> dict[str, Any]:
        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise ValueError("다음 전개 지시를 입력해줘.")
        if creativity_profile not in CREATIVITY_LEVELS:
            raise ValueError("지원하지 않는 창의성 단계야.")
        if requested_chars not in self.config.consumer.allowed_turn_chars:
            raise ValueError("한 턴 분량은 2,000자, 3,000자, 5,000자 중에서 골라줘.")

        now = iso_time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._maintenance_status(connection) != MAINTENANCE_OFF:
                raise MaintenanceModeError("지금은 점검 중이야. 잠시 뒤 다시 시도해줘.")
            story = connection.execute(
                "SELECT * FROM stories WHERE id = ? AND owner_id = ? AND deleted_at IS NULL",
                (story_id, owner_id),
            ).fetchone()
            if story is None:
                raise AuthorizationError("이 계정에서 열 수 없는 작품이야.")
            if story["expires_at"] <= now:
                raise ConsumerStoreError("보관 기간이 끝난 작품이야.")
            if story["completed_at"]:
                raise ConsumerStoreError("이미 결말까지 완성한 작품이야. 전체 원고를 내려받아 감상해줘.")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO jobs(
                        story_id, instruction, creativity_profile, requested_chars, status,
                        start_section_count, created_at
                    ) VALUES(?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (
                        story_id,
                        clean_instruction,
                        creativity_profile,
                        requested_chars,
                        int(story["section_count"]),
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StoryBusyError("이 작품은 이미 생성 요청을 처리하고 있어.") from exc
            job_id = int(cursor.lastrowid)
        job = self.get_job(job_id)
        assert job is not None
        return job

    def claim_next_job(self, worker_id: str, model_version: str) -> dict[str, Any] | None:
        now = iso_time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._maintenance_status(connection) == MAINTENANCE_ACTIVE:
                return None
            running = connection.execute(
                "SELECT 1 FROM jobs WHERE status = 'running' LIMIT 1"
            ).fetchone()
            if running is not None:
                return None
            row = connection.execute(
                """
                SELECT jobs.* FROM jobs
                JOIN stories ON stories.id = jobs.story_id
                WHERE jobs.status = 'queued' AND stories.deleted_at IS NULL
                ORDER BY jobs.created_at, jobs.id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running', worker_id = ?, model_version = ?,
                    started_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (worker_id, model_version, now, now, int(row["id"])),
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE id = ?", (int(row["id"]),)).fetchone()
        return _row_dict(claimed)

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_dict(row)

    def list_jobs(self, story_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE story_id = ? ORDER BY id DESC LIMIT ?",
                (story_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_owned_jobs(
        self,
        owner_id: str,
        story_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT jobs.* FROM jobs
                JOIN stories ON stories.id = jobs.story_id
                WHERE jobs.story_id = ? AND stories.owner_id = ? AND stories.deleted_at IS NULL
                ORDER BY jobs.id DESC LIMIT ?
                """,
                (story_id, owner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def outstanding_job(self, story_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE story_id = ? AND status IN ('queued', 'running')
                ORDER BY id DESC LIMIT 1
                """,
                (story_id,),
            ).fetchone()
        return _row_dict(row)

    def owned_outstanding_job(self, owner_id: str, story_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT jobs.* FROM jobs
                JOIN stories ON stories.id = jobs.story_id
                WHERE jobs.story_id = ? AND stories.owner_id = ?
                  AND stories.deleted_at IS NULL AND jobs.status IN ('queued', 'running')
                ORDER BY jobs.id DESC LIMIT 1
                """,
                (story_id, owner_id),
            ).fetchone()
        return _row_dict(row)

    def queue_position(self, owner_id: str, job_id: int) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT jobs.* FROM jobs
                JOIN stories ON stories.id = jobs.story_id
                WHERE jobs.id = ? AND stories.owner_id = ? AND stories.deleted_at IS NULL
                """,
                (job_id, owner_id),
            ).fetchone()
        job = _row_dict(row)
        if job is None or job["status"] != JOB_QUEUED:
            return None
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS ahead FROM jobs
                WHERE status = 'queued'
                  AND (created_at < ? OR (created_at = ? AND id < ?))
                """,
                (job["created_at"], job["created_at"], job_id),
            ).fetchone()
        return int(row["ahead"]) + 1

    def heartbeat_job(self, job_id: int, worker_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND worker_id = ? AND status = 'running'",
                (iso_time(), job_id, worker_id),
            )

    def sync_story_progress(self, story_id: str, total_chars: int, section_count: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE stories SET current_chars = ?, section_count = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (total_chars, section_count, iso_time(), story_id),
            )

    def set_job_start_section_count(self, job_id: int, section_count: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET start_section_count = ? WHERE id = ? AND status = 'running'",
                (section_count, job_id),
            )

    def complete_job(
        self,
        job_id: int,
        *,
        result_chars: int,
        result_section_count: int,
        total_chars: int,
        total_section_count: int,
        metrics: dict[str, Any],
        novel_completed: bool = False,
    ) -> None:
        now = iso_time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT story_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ConsumerStoreError("Generation job no longer exists.")
            connection.execute(
                """
                UPDATE jobs SET status = 'succeeded', result_chars = ?, result_section_count = ?,
                    metrics_json = ?, heartbeat_at = ?, finished_at = ?, error_public = NULL
                WHERE id = ? AND status = 'running'
                """,
                (
                    result_chars,
                    result_section_count,
                    json.dumps(metrics, ensure_ascii=False),
                    now,
                    now,
                    job_id,
                ),
            )
            connection.execute(
                """
                UPDATE stories SET current_chars = ?, section_count = ?, updated_at = ?,
                    completed_at = COALESCE(completed_at, ?)
                WHERE id = ? AND deleted_at IS NULL
                """,
                (
                    total_chars,
                    total_section_count,
                    now,
                    now if novel_completed else None,
                    row["story_id"],
                ),
            )
            self._advance_maintenance(connection)

    def fail_job(self, job_id: int, error_public: str, *, recoverable: bool) -> None:
        status = JOB_FAILED_RECOVERABLE if recoverable else JOB_FAILED
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = ?, error_public = ?, heartbeat_at = ?, finished_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (status, error_public[:1000], iso_time(), iso_time(), job_id),
            )
            self._advance_maintenance(connection)

    def save_section_metrics(
        self,
        *,
        story_id: str,
        job_id: int,
        model_version: str,
        creativity_profile: str,
        values: Sequence[tuple[int, dict[str, Any]]],
    ) -> None:
        now = iso_time()
        rows = [
            (
                story_id,
                job_id,
                section_index,
                model_version,
                creativity_profile,
                json.dumps(metrics, ensure_ascii=False),
                now,
            )
            for section_index, metrics in values
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO section_metrics(
                    story_id, job_id, section_index, model_version,
                    creativity_profile, metrics_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(story_id, section_index) DO UPDATE SET
                    job_id = excluded.job_id,
                    model_version = excluded.model_version,
                    creativity_profile = excluded.creativity_profile,
                    metrics_json = excluded.metrics_json,
                    created_at = excluded.created_at
                """,
                rows,
            )

    def recover_stale_jobs(self, stale_after_sec: int | None = None) -> int:
        seconds = stale_after_sec or self.config.consumer.stale_job_sec
        cutoff = iso_time(utc_now() - timedelta(seconds=seconds))
        now = iso_time()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed_recoverable',
                    error_public = '생성 worker가 중단됐어. 저장된 섹션은 보존되며 다시 이어 쓸 수 있어.',
                    finished_at = ?
                WHERE status = 'running' AND COALESCE(heartbeat_at, started_at, created_at) < ?
                """,
                (now, cutoff),
            )
            self._advance_maintenance(connection)
        return int(cursor.rowcount)

    def set_maintenance(self, enabled: bool) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if enabled:
                outstanding = connection.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')"
                ).fetchone()
                value = MAINTENANCE_DRAINING if int(outstanding["count"]) else MAINTENANCE_ACTIVE
            else:
                value = MAINTENANCE_OFF
            self._set_state_connection(connection, "maintenance", value)

    def is_maintenance(self) -> bool:
        return self.maintenance_status() != MAINTENANCE_OFF

    def maintenance_status(self) -> str:
        with self.connect() as connection:
            return self._maintenance_status(connection)

    def _maintenance_status(self, connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM service_state WHERE key = 'maintenance'"
        ).fetchone()
        return str(row["value"]) if row else MAINTENANCE_OFF

    def _advance_maintenance(self, connection: sqlite3.Connection) -> None:
        if self._maintenance_status(connection) != MAINTENANCE_DRAINING:
            return
        outstanding = connection.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
        if int(outstanding["count"]) == 0:
            self._set_state_connection(connection, "maintenance", MAINTENANCE_ACTIVE)

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as connection:
            self._set_state_connection(connection, key, value)

    def _set_state_connection(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO service_state(key, value, updated_at) VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, iso_time()),
        )

    def get_state(self, key: str) -> dict[str, str] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value, updated_at FROM service_state WHERE key = ?",
                (key,),
            ).fetchone()
        return dict(row) if row else None

    def heartbeat_worker(self, worker_id: str, status: str = "idle") -> None:
        payload = json.dumps({"worker_id": worker_id, "status": status}, ensure_ascii=False)
        self.set_state("worker_heartbeat", payload)

    def queue_stats(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "queued": counts.get(JOB_QUEUED, 0),
            "running": counts.get(JOB_RUNNING, 0),
            "succeeded": counts.get(JOB_SUCCEEDED, 0),
            "failed": counts.get(JOB_FAILED, 0),
            "failed_recoverable": counts.get(JOB_FAILED_RECOVERABLE, 0),
        }

    def anonymous_metric_rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT section_metrics.model_version, section_metrics.creativity_profile,
                       section_metrics.metrics_json
                FROM section_metrics
                JOIN stories ON stories.id = section_metrics.story_id
                WHERE stories.deleted_at IS NULL
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "model_version": row["model_version"],
                    "creativity_profile": row["creativity_profile"],
                    **json.loads(row["metrics_json"]),
                }
            )
        return result

    def consented_stories(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, genre, current_chars, section_count, created_at, updated_at
                FROM stories
                WHERE research_consent = 1 AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_expired(self) -> int:
        now = iso_time()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM stories WHERE expires_at <= ? AND deleted_at IS NULL",
                (now,),
            ).fetchall()
        for row in rows:
            self.delete_story(str(row["id"]))
        return len(rows)

    def delete_owned_story(self, owner_id: str, story_id: str) -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM stories WHERE id = ? AND owner_id = ? AND deleted_at IS NULL",
                (story_id, owner_id),
            ).fetchone()
        if row is None:
            raise AuthorizationError("이 계정에서 삭제할 수 없는 작품이야.")
        self.delete_story(story_id)

    def delete_story(self, story_id: str) -> None:
        workspace = StoryWorkspace.for_story(self.config, story_id)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM stories WHERE id = ?", (story_id,))
        workspace.delete()

    def _validated_story_fields(
        self,
        *,
        title: str,
        genre: str,
        premise: str,
        world: str,
        protagonist: str,
        characters: str,
        target_chars: int,
        floor_chars: int = 0,
    ) -> dict[str, Any]:
        fields = {
            "title": title.strip(),
            "genre": genre.strip(),
            "premise": premise.strip(),
            "world": world.strip(),
            "protagonist": protagonist.strip(),
            "characters": characters.strip(),
        }
        if any(not fields[name] for name in ("title", "genre", "premise", "world", "protagonist")):
            raise ValueError("제목, 장르, 소재, 세계관, 주인공을 모두 입력해줘.")
        target = int(target_chars)
        if target < self.config.consumer.min_target_chars:
            raise ValueError(
                f"전체 목표 분량은 {self.config.consumer.min_target_chars:,}자 이상이어야 해."
            )
        if target > self.config.consumer.max_target_chars:
            raise ValueError(
                f"전체 목표 분량은 {self.config.consumer.max_target_chars:,}자 이하여야 해."
            )
        if target < floor_chars:
            raise ValueError(
                f"이미 {floor_chars:,}자를 썼어. 목표는 그보다 작을 수 없어."
            )
        fields["target_chars"] = target
        return fields

    def update_owned_story(
        self,
        owner_id: str,
        story_id: str,
        *,
        title: str,
        genre: str,
        premise: str,
        world: str,
        protagonist: str,
        characters: str = "",
        target_chars: int,
        research_consent: bool,
    ) -> dict[str, Any]:
        """Edit the settings a story was created with.

        The world and character sheets are rebuilt into every section prompt, so
        an edit changes the canon from the next turn onward. Sections already
        written are left alone; rewriting them would invalidate the memory and
        ledger that later sections were built on.
        """
        story = self.get_owned_story(owner_id, story_id)
        if story is None:
            raise AuthorizationError("이 계정에서 수정할 수 없는 작품이야.")
        fields = self._validated_story_fields(
            title=title,
            genre=genre,
            premise=premise,
            world=world,
            protagonist=protagonist,
            characters=characters,
            target_chars=target_chars,
            floor_chars=int(story["current_chars"]),
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE stories
                SET title = ?, genre = ?, premise = ?, world = ?, protagonist = ?,
                    characters = ?, target_chars = ?, research_consent = ?, updated_at = ?
                WHERE id = ? AND owner_id = ? AND deleted_at IS NULL
                """,
                (
                    fields["title"],
                    fields["genre"],
                    fields["premise"],
                    fields["world"],
                    fields["protagonist"],
                    fields["characters"],
                    fields["target_chars"],
                    1 if research_consent else 0,
                    iso_time(),
                    story_id,
                    owner_id,
                ),
            )
        updated = self.get_owned_story(owner_id, story_id)
        if updated is None:
            raise AuthorizationError("이 계정에서 수정할 수 없는 작품이야.")
        return updated

    def reset_owned_story(self, owner_id: str, story_id: str) -> None:
        """Throw away the manuscript but keep the story and its settings.

        Rows go, not flags: the jobs are deleted outright and the progress
        counters return to zero, so the story is indistinguishable from one that
        was just created.
        """
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                "SELECT 1 FROM stories WHERE id = ? AND owner_id = ? AND deleted_at IS NULL",
                (story_id, owner_id),
            ).fetchone()
            if owned is None:
                raise AuthorizationError("이 계정에서 초기화할 수 없는 작품이야.")
            placeholders = ",".join("?" for _ in OUTSTANDING_JOB_STATUSES)
            active = connection.execute(
                f"SELECT 1 FROM jobs WHERE story_id = ? AND status IN ({placeholders})",
                (story_id, *OUTSTANDING_JOB_STATUSES),
            ).fetchone()
            if active is not None:
                raise StoryBusyError("집필이 끝난 뒤에 초기화할 수 있어.")
            connection.execute("DELETE FROM jobs WHERE story_id = ?", (story_id,))
            connection.execute(
                """
                UPDATE stories
                SET current_chars = 0, section_count = 0, completed_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (iso_time(), story_id),
            )
        StoryWorkspace.for_story(self.config, story_id).reset()

    def delete_owned_job(self, owner_id: str, story_id: str, job_id: int) -> None:
        """Remove one finished turn from the conversation log.

        This deletes the chat record only. Prose the turn produced already lives
        in the manuscript, and pulling a section back out would break the memory,
        ledger, and outline that later sections were written against.
        """
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT jobs.status FROM jobs
                JOIN stories ON stories.id = jobs.story_id
                WHERE jobs.id = ? AND jobs.story_id = ?
                  AND stories.owner_id = ? AND stories.deleted_at IS NULL
                """,
                (job_id, story_id, owner_id),
            ).fetchone()
            if row is None:
                raise AuthorizationError("이 계정에서 삭제할 수 없는 대화야.")
            if str(row["status"]) in OUTSTANDING_JOB_STATUSES:
                raise ConsumerStoreError("집필 중이거나 대기 중인 요청은 지울 수 없어.")
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def clear_owned_job_history(self, owner_id: str, story_id: str) -> int:
        """Delete every finished turn from the log, keeping active ones."""
        placeholders = ",".join("?" for _ in OUTSTANDING_JOB_STATUSES)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                "SELECT 1 FROM stories WHERE id = ? AND owner_id = ? AND deleted_at IS NULL",
                (story_id, owner_id),
            ).fetchone()
            if owned is None:
                raise AuthorizationError("이 계정에서 삭제할 수 없는 작품이야.")
            cursor = connection.execute(
                f"DELETE FROM jobs WHERE story_id = ? AND status NOT IN ({placeholders})",
                (story_id, *OUTSTANDING_JOB_STATUSES),
            )
        return int(cursor.rowcount or 0)
