from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from src.evaluation.metrics import (
    contradiction_check,
    cosine_similarity,
    dialogue_ratio,
    hallucination_metrics,
    keyword_consistency,
    length_fit,
    lexical_diversity,
    novelty_from_previous,
    overall_score,
    pairwise_diversity,
    progression_score,
    repetition_profile,
    repetition_rate,
    section_structure_metrics,
    sentence_stats,
)
from src.evaluation.llm_judge import JUDGE_SCORE_KEYS, JUDGE_SCORE_LABELS, run_llm_judge
from src.generation.consistency import check_name_consistency
from src.llm.ollama_client import OllamaClient
from src.memory.beat_ledger import narrative_beat_metrics
from src.planner.predict import evaluate_planner_diagnostics
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def _embedding_excerpt(text: str, max_chars: int = 6000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return "\n\n[...middle omitted for embedding...]\n\n".join([cleaned[:half], cleaned[-half:]])


def _embedding_map(client: OllamaClient, previous_scene: str, outputs: dict[str, str]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    nonempty = [(name, text) for name, text in outputs.items() if text.strip()]
    if not previous_scene.strip() or not nonempty:
        return None, {}
    vectors = client.embed([_embedding_excerpt(previous_scene)] + [_embedding_excerpt(text) for _name, text in nonempty])
    return vectors[0], {name: vector for (name, _text), vector in zip(nonempty, vectors[1:])}


def evaluate_outputs(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    outputs: dict[str, str],
    characters: str = "",
    world: str = "",
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
            **section_structure_metrics(
                text,
                expected_sections=config.generation.section_count,
                min_chars_per_section=config.generation.section_min_chars,
            ),
            **narrative_beat_metrics(text),
            **sentence_stats(text),
        }
        row.update(
            hallucination_metrics(
                previous_scene,
                text,
                target_novelty=config.generation.hallucination_target,
                contradictions=row["contradictions"],
                name_consistency_score=consistency.score,
            )
        )
        row["overall_score"] = overall_score(row)
        rows[name] = row
    if config.evaluation.use_llm_judge:
        # The judge is diagnostic only: it never feeds overall_score or the
        # ranking, so enabling it cannot reorder existing comparisons.
        for name, text in outputs.items():
            rows[name]["llm_judge"] = run_llm_judge(
                client,
                previous_scene,
                text,
                world=world,
                characters=characters,
                temperature=config.evaluation.judge_temperature,
                max_tokens=config.evaluation.judge_max_tokens,
                excerpt_chars=config.evaluation.judge_excerpt_chars,
            )
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
    metrics = evaluate_outputs(config, client, previous_scene, outputs, characters=characters, world=world)
    run_state_path = resolve_path(config, config.generation.longform_state_path)
    generation_guard = {
        "repetition_retry_count": 0,
        "retry_success_count": 0,
        "retry_success_rate": 0.0,
    }
    if run_state_path.exists():
        try:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            retry_count = int(run_state.get("repetition_retry_count", 0) or 0)
            success_count = int(run_state.get("retry_success_count", 0) or 0)
            generation_guard = {
                "repetition_retry_count": retry_count,
                "retry_success_count": success_count,
                "retry_success_rate": round(success_count / max(1, retry_count), 4),
                "recent_retry_reasons": run_state.get("retry_reasons", []),
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    metrics["generation_guard"] = generation_guard
    plausibility = {
        "enabled": bool(config.generation.enable_jepa_coherence_gate),
        "threshold": float(config.generation.jepa_coherence_min_cosine),
        "scored_sections": 0,
        "mean_jepa_coherence": 0.0,
        "min_jepa_coherence": 0.0,
        "coherence_retry_count": 0,
        "coherence_retry_success_count": 0,
    }
    if run_state_path.exists():
        try:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            plausibility.update(
                {
                    "scored_sections": int(run_state.get("jepa_coherence_scored_sections", 0) or 0),
                    "mean_jepa_coherence": float(run_state.get("mean_jepa_coherence", 0.0) or 0.0),
                    "min_jepa_coherence": float(run_state.get("min_jepa_coherence", 0.0) or 0.0),
                    "coherence_retry_count": int(run_state.get("coherence_retry_count", 0) or 0),
                    "coherence_retry_success_count": int(
                        run_state.get("coherence_retry_success_count", 0) or 0
                    ),
                }
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    metrics["jepa_plausibility_gate"] = plausibility
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
        f"- validation_pred_target_cosine: {planner_diagnostics.get('validation_pred_target_cosine', 0.0):.4f}",
        f"- validation_retrieval_hit_at_k: {planner_diagnostics.get('validation_retrieval_hit_at_k', 0.0):.4f}",
        f"- validation_retrieval_mean_score: {planner_diagnostics.get('validation_retrieval_mean_score', 0.0):.4f}",
        f"- validation_transition_direction_diversity: {planner_diagnostics.get('validation_transition_direction_diversity', 0.0):.4f}",
        "",
        "### Narrative Plausibility Gate",
        "",
        "The trained predictor also scores finished prose: each section's realized next-state "
        "embedding is compared with the state predicted from the preceding context. Low-cosine "
        "sections are rewritten once. The threshold is specific to this embedding model and "
        "predictor; re-measure it with `scripts/calibrate_coherence.py`.",
        "",
        f"- gate enabled: {plausibility['enabled']}",
        f"- threshold: {plausibility['threshold']:.3f}",
        f"- scored sections: {plausibility['scored_sections']}",
        f"- mean_jepa_coherence: {plausibility['mean_jepa_coherence']:.4f}",
        f"- min_jepa_coherence: {plausibility['min_jepa_coherence']:.4f}",
        f"- coherence rewrites: {plausibility['coherence_retry_count']} "
        f"(resolved {plausibility['coherence_retry_success_count']})",
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
        "## Controlled Hallucination Metrics",
        "",
        "Creative expansion is the ratio of generated content tokens not seen in the previous scene. "
        "Useful hallucination rewards novel material that remains consistent and advances the scene.",
        "",
    ]
    for name, row in metrics["modes"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- creative_expansion_rate: {row.get('creative_expansion_rate', 0.0):.4f}",
                f"- hallucination_presence: {row.get('hallucination_presence', 'none')}",
                f"- hallucination_target_alignment: {row.get('hallucination_target_alignment', 0.0):.4f}",
                f"- useful_hallucination_score: {row.get('useful_hallucination_score', 0.0):.4f}",
                f"- hallucination_risk: {row.get('hallucination_risk', 0.0):.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## LLM Judge",
            "",
            "An optional local-model review of each output. Scores are 1-10 per rubric axis and "
            "diagnostic only: they never change overall_score or the ranking. "
            "`hallucination_control` separates invented detail that enriches the story from "
            "invented detail that contradicts the established setup.",
            "",
        ]
    )
    if not config.evaluation.use_llm_judge:
        lines.extend(["- disabled (`evaluation.use_llm_judge: false`)", ""])
    else:
        for name, row in metrics["modes"].items():
            judge = row.get("llm_judge", {})
            lines.extend([f"### {name}", ""])
            if not judge.get("available"):
                lines.extend(
                    [
                        f"- unavailable: {judge.get('error', 'no result')}",
                        "",
                    ]
                )
                continue
            scores = judge.get("scores", {})
            for key in JUDGE_SCORE_KEYS:
                if key in scores:
                    lines.append(f"- {key} ({JUDGE_SCORE_LABELS[key]}): {scores[key]:.1f} / 10")
            lines.append(f"- judge_overall: {judge.get('overall', 0.0):.4f}")
            if judge.get("verdict"):
                lines.append(f"- verdict: {judge['verdict']}")
            useful = judge.get("useful_hallucinations", [])
            if useful:
                lines.extend(["", "Useful hallucinations:", ""])
                lines.extend(f"- {item}" for item in useful)
            harmful = judge.get("harmful_hallucinations", [])
            if harmful:
                lines.extend(["", "Harmful hallucinations:", ""])
                lines.extend(f"- {item}" for item in harmful)
            lines.append("")
    lines.extend(
        [
            "## Narrative Repetition Guard",
            "",
            "Consumed-beat metrics distinguish repeated plot events from ordinary word-level repetition.",
            "",
            f"- repetition_retry_count: {generation_guard['repetition_retry_count']}",
            f"- retry_success_count: {generation_guard['retry_success_count']}",
            f"- retry_success_rate: {generation_guard['retry_success_rate']:.4f}",
            "",
        ]
    )
    for name, row in metrics["modes"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- repeated_subtitle_count: {row.get('repeated_subtitle_count', 0)}",
                f"- repeated_narrative_beat_count: {row.get('repeated_narrative_beat_count', 0)}",
                f"- adjacent_section_similarity: {row.get('adjacent_section_similarity', 0.0):.4f}",
                f"- max_adjacent_section_similarity: {row.get('max_adjacent_section_similarity', 0.0):.4f}",
                "",
            ]
        )
    lines.extend([
        "## Section Structure Metrics",
        "",
        "These metrics check whether the output is shaped like a sectioned novel scene with titled parts and substantial body text.",
        "",
    ])
    for name, row in metrics["modes"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- section_count: {row.get('section_count', 0)}",
                f"- section_count_fit: {row.get('section_count_fit', 0.0):.4f}",
                f"- sections_with_body: {row.get('sections_with_body', 0)}",
                f"- section_body_coverage: {row.get('section_body_coverage', 0.0):.4f}",
                f"- avg_section_body_chars: {row.get('avg_section_body_chars', 0.0):.2f}",
                "",
            ]
        )
    lines.extend([
        "## Consistency Findings",
        "",
    ])
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
