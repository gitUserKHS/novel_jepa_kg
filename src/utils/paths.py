from __future__ import annotations

from pathlib import Path

from src.utils.config import AppConfig


def project_root(config: AppConfig) -> Path:
    return Path(config.output_root).expanduser().resolve()


def resolve_path(config: AppConfig, relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return project_root(config) / path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_project_dirs(config: AppConfig) -> None:
    for directory in [
        "data/raw",
        "data/synthetic",
        "data/filtered",
        "data/embeddings",
        "data/indexes",
        config.chat.session_dir,
        "checkpoints/predictor",
        config.evaluation.report_dir,
    ]:
        resolve_path(config, directory).mkdir(parents=True, exist_ok=True)
