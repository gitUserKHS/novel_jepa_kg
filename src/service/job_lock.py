from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from src.utils.config import AppConfig
from src.utils.paths import resolve_path


class ServiceBusyError(RuntimeError):
    """Raised when another session owns the shared artifact lock."""


def _lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _metadata(handle: BinaryIO) -> dict[str, object]:
    try:
        handle.seek(1)
        return json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


class ProjectJobLease:
    def __init__(self, path: Path, handle: BinaryIO, operation: str) -> None:
        self.path = path
        self.operation = operation
        self._handle = handle
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()
            self._released = True

    def __enter__(self) -> ProjectJobLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def acquire_project_job(config: AppConfig, operation: str) -> ProjectJobLease:
    path = resolve_path(config, config.service.job_lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
        try:
            _lock(handle)
        except OSError as exc:
            current = _metadata(handle)
            owner = current.get("operation", "another operation")
            started = current.get("started_at", "unknown time")
            raise ServiceBusyError(
                f"Shared artifacts are busy: {owner} started at {started}. Try again after it finishes."
            ) from exc

        metadata = {
            "operation": operation,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps(metadata, ensure_ascii=True).encode("utf-8"))
        return ProjectJobLease(path, handle, operation)
    except Exception:
        handle.close()
        raise
