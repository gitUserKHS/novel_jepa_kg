from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def session_dir(config: AppConfig) -> Path:
    path = resolve_path(config, config.chat.session_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_path(config: AppConfig) -> Path:
    return session_dir(config) / "index.json"


def safe_session_id(title: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", title.strip()).strip("-")
    slug = slug[:40] or "session"
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}_{uuid.uuid4().hex[:8]}"


def session_path(config: AppConfig, session_id: str) -> Path:
    return session_dir(config) / f"{session_id}.json"


def default_session(title: str, world: str = "", characters: str = "") -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "session_id": safe_session_id(title),
        "title": title.strip() or "Untitled session",
        "created_at": timestamp,
        "updated_at": timestamp,
        "world": world.strip(),
        "characters": characters.strip(),
        "messages": [],
        "scene_summaries": [],
        "memory_summary": "",
        "knowledge_graph": {"nodes": [], "edges": []},
        "story_state": {
            "current_scene": "",
            "major_goal": "",
            "unresolved_hooks": [],
            "style_notes": [],
            "warnings": [],
        },
    }


def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "title": session.get("title", "Untitled session"),
        "updated_at": session.get("updated_at", ""),
        "created_at": session.get("created_at", ""),
        "message_count": len(session.get("messages", [])),
        "scene_count": len(session.get("scene_summaries", [])),
    }


def write_index(config: AppConfig, sessions: list[dict[str, Any]]) -> None:
    summaries = [_session_summary(session) for session in sessions]
    summaries.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
    index_path(config).write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def list_sessions(config: AppConfig) -> list[dict[str, Any]]:
    sessions = []
    for path in session_dir(config).glob("*.json"):
        if path.name == "index.json":
            continue
        try:
            sessions.append(_session_summary(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    sessions.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
    index_path(config).write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    return sessions


def load_session(config: AppConfig, session_id: str) -> dict[str, Any]:
    path = session_path(config, session_id)
    if not path.exists():
        raise FileNotFoundError(f"Chat session not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(config: AppConfig, session: dict[str, Any]) -> dict[str, Any]:
    session["updated_at"] = now_iso()
    path = session_path(config, session["session_id"])
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    all_sessions = []
    for summary in list_sessions(config):
        try:
            all_sessions.append(load_session(config, summary["session_id"]))
        except FileNotFoundError:
            continue
    if session["session_id"] not in {item["session_id"] for item in all_sessions}:
        all_sessions.append(session)
    write_index(config, all_sessions)
    return session


def create_session(config: AppConfig, title: str, world: str = "", characters: str = "") -> dict[str, Any]:
    session = default_session(title, world, characters)
    return save_session(config, session)


def delete_session(config: AppConfig, session_id: str) -> None:
    path = session_path(config, session_id)
    if path.exists():
        path.unlink()
    list_sessions(config)


def append_message(
    session: dict[str, Any],
    role: str,
    content: str,
    mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = {
        "role": role,
        "content": content,
        "created_at": now_iso(),
        "mode": mode,
        "metadata": metadata or {},
    }
    session.setdefault("messages", []).append(message)
    return message


def append_scene_summary(session: dict[str, Any], summary: str, mode: str) -> dict[str, Any]:
    scene = {
        "index": len(session.get("scene_summaries", [])) + 1,
        "summary": summary,
        "mode": mode,
        "created_at": now_iso(),
    }
    session.setdefault("scene_summaries", []).append(scene)
    session.setdefault("story_state", {})["current_scene"] = summary
    return scene


def export_session_markdown(config: AppConfig, session: dict[str, Any]) -> str:
    report_dir = resolve_path(config, config.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"session_{session['session_id']}.md"
    lines = [
        f"# {session.get('title', 'Untitled session')}",
        "",
        f"- Session ID: {session['session_id']}",
        f"- Updated: {session.get('updated_at', '')}",
        "",
        "## World",
        "",
        session.get("world", "") or "(empty)",
        "",
        "## Characters",
        "",
        session.get("characters", "") or "(empty)",
        "",
        "## Memory Summary",
        "",
        session.get("memory_summary", "") or "(empty)",
        "",
        "## Scenes",
    ]
    for scene in session.get("scene_summaries", []):
        lines.extend(["", f"### Scene {scene.get('index')}", "", scene.get("summary", "")])
    lines.extend(["", "## Messages"])
    for message in session.get("messages", []):
        lines.extend(["", f"### {message.get('role', '').title()} - {message.get('created_at', '')}", "", message.get("content", "")])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
