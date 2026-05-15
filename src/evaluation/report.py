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
from src.generation.consistency import check_name_consistency
from src.llm.ollama_client import OllamaClient
from src.planner.predict import evaluate_planner_diagnostics
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def _embedding_map(client: OllamaClient, previous_scene: str, outputs: dict[str, str]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    nonempty = [(name, text) for name, text in outputs.items() if text.strip()]
    if not previous_scene.strip() or not nonempty:
        return None, {}
    vectors = client.embed([previous_scene] + [text for _name, text in nonempty])
    return vectors[0], {name: vector for (name, _text), vector in zip(nonempty, vectors[1:])}


def evaluate_outputs(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    outputs: dict[str, str],
    characters: str = "",
) -> dict:
    reference_vector, output_vectors = _embedding_map(client, previous_scene, outputs)
    rows = {}
    for name, text in outputs.items():
        vector = output_vectors.get(name)
        embedding_continuity = cosine_similarity(reference_vector, vector) if reference_vector is not None and vector is not None else 0.0
        consistency = check_name_consistency(text, characters)
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
            "known_names": consistency.known_names,
            "name_consistency_issues": consistency.issues,
            "name_consistency_score": consistency.score,
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
    world: str = "",
    characters: str = "",
) -> str:
    report_dir = resolve_path(config, config.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_outputs(config, client, previous_scene, outputs, characters=characters)
    planner_diagnostics = evaluate_planner_diagnostics(config, top_k=config.generation.top_k)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"comparison_{timestamp}.md"
    lines = [
        "# Novel JEPA Lab Comparison Report",
        "",
        f"- Created: {timestamp}",
        f"- Previous scene: {previous_scene}",
        f"- World: {world or '(not provided)'}",
        f"- Characters: {characters or '(not provided)'}",
        f"- Ranking: {', '.join(metrics['ranking'])}",
        f"- Pairwise output diversity: {metrics['pairwise_output_diversity']:.4f}",
        "",
        "## Planner Diagnostics",
        "",
        "```json",
        json.dumps(planner_diagnostics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Consistency Findings",
        "",
    ]
    for name, row in metrics["modes"].items():
        issues = row.get("name_consistency_issues", [])
        lines.extend([f"### {name}", ""])
        if issues:
            lines.extend([f"- {issue}" for issue in issues])
        else:
            lines.append("- No name consistency issues detected.")
        lines.append("")
    lines.extend([
        "## Outputs",
    ])
    for name, text in outputs.items():
        lines.extend(["", f"### {name}", "", text or "(empty)"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
