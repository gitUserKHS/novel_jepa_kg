from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.health import model_is_available
from src.service.artifacts import active_model_status
from src.service.consumer_store import ConsumerStore
from src.utils.config import load_config


def _retry(label: str, attempts: int, delay: float, check: Callable[[], Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return check()
        except Exception as exc:  # noqa: BLE001 - the final error keeps the probe context.
            last_error = exc
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def check_streamlit(app_url: str, timeout: float) -> None:
    response = requests.get(f"{app_url.rstrip('/')}/_stcore/health", timeout=timeout)
    response.raise_for_status()
    if response.text.strip().lower() != "ok":
        raise RuntimeError(f"Unexpected Streamlit health response: {response.text[:120]!r}")


def check_ollama(ollama_url: str, models_required: list[str], timeout: float) -> list[str]:
    response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    models = [item.get("name", "") for item in response.json().get("models", [])]
    missing = [model for model in models_required if not model_is_available(model, models)]
    if missing:
        raise RuntimeError(f"Ollama models are missing: {missing}; installed models: {models}")
    return models


def check_worker(config_path: str, max_age_sec: float) -> dict[str, Any]:
    store = ConsumerStore(load_config(config_path))
    state = store.get_state("worker_heartbeat")
    if state is None:
        raise RuntimeError("Consumer worker has not written a heartbeat.")
    updated = datetime.fromisoformat(state["updated_at"])
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - updated).total_seconds()
    if age > max_age_sec:
        raise RuntimeError(f"Consumer worker heartbeat is stale ({age:.1f}s).")
    return json.loads(state["value"])


def check_active_jepa(config_path: str) -> str:
    status = active_model_status(load_config(config_path), verify_files=True)
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    return str(status["manifest"]["version"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local Novel JEPA service and Ollama runtime.")
    parser.add_argument("--app-url", default="http://127.0.0.1:8501")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--embedding-model", default="embeddinggemma:latest")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--worker-max-age", type=float, default=20.0)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-worker", action="store_true")
    parser.add_argument("--skip-active-jepa", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _retry(
            "Streamlit health check",
            args.attempts,
            args.delay,
            lambda: check_streamlit(args.app_url, args.timeout),
        )
        if not args.skip_ollama:
            _retry(
                "Ollama health check",
                args.attempts,
                args.delay,
                lambda: check_ollama(
                    args.ollama_url,
                    [args.model, args.embedding_model],
                    args.timeout,
                ),
            )
        if not args.skip_worker:
            _retry(
                "Consumer worker heartbeat",
                args.attempts,
                args.delay,
                lambda: check_worker(args.config, args.worker_max_age),
            )
        if not args.skip_active_jepa:
            _retry(
                "Active JEPA artifact health",
                args.attempts,
                args.delay,
                lambda: check_active_jepa(args.config),
            )
    except Exception as exc:  # noqa: BLE001 - command-line probes report one clear error.
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] Novel JEPA service is healthy at {args.app_url}")
    if not args.skip_ollama:
        print(f"[OK] Ollama models are available: {args.model}, {args.embedding_model}")
    if not args.skip_worker:
        print("[OK] Consumer worker heartbeat is fresh")
    if not args.skip_active_jepa:
        print("[OK] Active JEPA artifact fingerprints match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
