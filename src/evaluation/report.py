from __future__ import annotations

import json
from datetime import datetime

from src.evaluation.metrics import contradiction_check, embedding_continuity, keyword_consistency, repetition_rate
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def evaluate_outputs(config: AppConfig, client: OllamaClient, previous_scene: str, outputs: dict[str, str]) -> dict:
    rows = {}
    for name, text in outputs.items():
        rows[name] = {
            "repetition_rate": repetition_rate(text, config.evaluation.repetition_ngram),
            "embedding_continuity": embedding_continuity(client, previous_scene, text) if text.strip() else 0.0,
            "keyword_consistency": keyword_consistency(previous_scene, text),
            "contradictions": contradiction_check(text),
        }
    return rows


def evaluate_and_write_report(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    outputs: dict[str, str],
) -> str:
    report_dir = resolve_path(config, config.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_outputs(config, client, previous_scene, outputs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"comparison_{timestamp}.md"
    lines = [
        "# Novel JEPA Lab Comparison Report",
        "",
        f"- Created: {timestamp}",
        f"- Previous scene: {previous_scene}",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Outputs",
    ]
    for name, text in outputs.items():
        lines.extend(["", f"### {name}", "", text or "(empty)"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
