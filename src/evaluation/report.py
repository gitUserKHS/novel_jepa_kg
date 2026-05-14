from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from src.evaluation.metrics import (
    contradiction_check,
    cosine_similarity,
    dialogue_ratio,
    keyword_consistency,
    length_fit,
    lexical_diversity,
    novelty_from_previous,
    overall_score,
    pairwise_diversity,
    progression_score,
    repetition_profile,
    repetition_rate,
    sentence_stats,
)
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def _embedding_map(client: OllamaClient, previous_scene: str, outputs: dict[str, str]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    nonempty = [(name, text) for name, text in outputs.items() if text.strip()]
    if not previous_scene.strip() or not nonempty:
        return None, {}
    vectors = client.embed([previous_scene] + [text for _name, text in nonempty])
    return vectors[0], {name: vector for (name, _text), vector in zip(nonempty, vectors[1:])}


def evaluate_outputs(config: AppConfig, client: OllamaClient, previous_scene: str, outputs: dict[str, str]) -> dict:
    reference_vector, output_vectors = _embedding_map(client, previous_scene, outputs)
    rows = {}
    for name, text in outputs.items():
        vector = output_vectors.get(name)
        embedding_continuity = cosine_similarity(reference_vector, vector) if reference_vector is not None and vector is not None else 0.0
        row = {
            "char_count": len(text.strip()),
            "repetition_rate": repetition_rate(text, config.evaluation.repetition_ngram),
            "repetition_profile": repetition_profile(text),
            "embedding_continuity": embedding_continuity,
            "keyword_consistency": keyword_consistency(previous_scene, text),
            "novelty_from_previous": novelty_from_previous(previous_scene, text),
            "lexical_diversity": lexical_diversity(text),
            "dialogue_ratio": dialogue_ratio(text),
            "length_fit": length_fit(text, config.evaluation.target_min_chars, config.evaluation.target_max_chars),
            "progression_score": progression_score(text),
            "contradictions": contradiction_check(text),
            **sentence_stats(text),
        }
        row["overall_score"] = overall_score(row)
        rows[name] = row
    ranking = sorted(rows, key=lambda key: rows[key]["overall_score"], reverse=True)
    return {
        "modes": rows,
        "ranking": ranking,
        "pairwise_output_diversity": pairwise_diversity(output_vectors),
    }


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
        f"- Ranking: {', '.join(metrics['ranking'])}",
        f"- Pairwise output diversity: {metrics['pairwise_output_diversity']:.4f}",
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
