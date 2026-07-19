from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.consumer_store import ConsumerStore
from src.utils.config import load_config


def snapshot(store: ConsumerStore) -> dict[str, object]:
    return {
        "maintenance": store.maintenance_status(),
        "queue": store.queue_stats(),
        "worker": store.get_state("worker_heartbeat"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Control Novel JEPA consumer maintenance state.")
    parser.add_argument("command", choices=["enter", "resume", "wait-idle", "status"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()
    store = ConsumerStore(load_config(args.config))
    if args.command == "enter":
        store.set_maintenance(True)
    elif args.command == "resume":
        store.set_maintenance(False)
    elif args.command == "wait-idle":
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            state = snapshot(store)
            queue = state["queue"]
            if state["maintenance"] == "1" and not queue["queued"] and not queue["running"]:
                print(json.dumps(state, ensure_ascii=False))
                return 0
            time.sleep(max(0.2, args.poll))
        print(json.dumps(snapshot(store), ensure_ascii=False))
        return 2
    print(json.dumps(snapshot(store), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
