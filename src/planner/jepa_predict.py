from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.embedding.vector_store import retrieve_next_by_vector
from src.llm.ollama_client import OllamaClient
from src.planner.jepa_dataset import build_generation_context_text
from src.planner.scene_analyzer import analyze_current_scene, build_analyzed_generation_context
from src.planner.jepa_train import load_predictor
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype("float32")


def _predict_from_embeddings(config: AppConfig, context_embeddings: np.ndarray) -> np.ndarray:
    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    model = load_predictor(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    try:
        with torch.no_grad():
            inputs = torch.tensor(context_embeddings, dtype=torch.float32, device=device)
            if device.type == "cuda" and inputs.shape[0] == 1:
                inputs = inputs.repeat(2, 1)
            pred = model(inputs).detach().cpu().numpy()[: len(context_embeddings)]
    except RuntimeError:
        if device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        model = load_predictor(checkpoint_path).cpu()
        with torch.no_grad():
            pred = model(torch.tensor(context_embeddings, dtype=torch.float32)).detach().cpu().numpy()
    return pred.astype("float32")


def _generation_context(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    world: str,
    characters: str,
    scene_preset: dict[str, str] | None,
) -> tuple[str, dict[str, Any] | None]:
    if config.generation.use_scene_analyzer:
        analysis = analyze_current_scene(config, client, world, characters, previous_scene, scene_preset=scene_preset)
        return build_analyzed_generation_context(world, characters, previous_scene, analysis, scene_preset=scene_preset), analysis
    return build_generation_context_text(world, characters, previous_scene, scene_preset=scene_preset), None


def predict_next_embedding(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    world: str = "",
    characters: str = "",
    scene_preset: dict[str, str] | None = None,
) -> np.ndarray:
    context_text, _analysis = _generation_context(config, client, previous_scene, world, characters, scene_preset)
    context = client.embed([context_text])
    return _predict_from_embeddings(config, context)[0].astype("float32")


def predict_next_embedding_with_diagnostics(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    world: str = "",
    characters: str = "",
    scene_preset: dict[str, str] | None = None,
) -> dict[str, Any]:
    context_text, analysis = _generation_context(config, client, previous_scene, world, characters, scene_preset)
    context = client.embed([context_text])
    predicted = _predict_from_embeddings(config, context)[0].astype("float32")
    retrieved = retrieve_next_by_vector(config, predicted, config.generation.top_k)
    return {
        "context_text": context_text,
        "analyzed_scene": analysis,
        "predicted_embedding": predicted,
        "predicted_vector_norm": float(np.linalg.norm(predicted)),
        "retrieved": retrieved,
        "retrieval_mean_score": float(np.mean([item["score"] for item in retrieved])) if retrieved else 0.0,
    }


def _topk(similarity: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, similarity.shape[1]))
    idx = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    row = np.arange(similarity.shape[0])[:, None]
    scores = similarity[row, idx]
    order = np.argsort(-scores, axis=1)
    return idx[row, order], scores[row, order]


def _retrieval_metrics(
    query_vectors: np.ndarray,
    target_vectors: np.ndarray,
    top_k: int,
    true_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    query = _normalize(query_vectors)
    target = _normalize(target_vectors)
    similarity = query @ target.T
    indices, scores = _topk(similarity, top_k)
    expected = true_indices if true_indices is not None else np.arange(indices.shape[0])
    hits = [int(int(expected[row_idx]) in set(row.tolist())) for row_idx, row in enumerate(indices)]
    unique_targets = len({int(idx) for row in indices for idx in row.tolist()})
    return {
        "retrieval_hit_at_k": float(np.mean(hits)) if hits else 0.0,
        "retrieval_mean_score": float(np.mean(scores)) if scores.size else 0.0,
        "transition_direction_diversity": float(unique_targets / max(1, indices.size)),
        "top_indices": indices,
        "top_scores": scores,
    }


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    overlaps = []
    for left_row, right_row in zip(left, right):
        left_set = set(int(item) for item in left_row.tolist())
        right_set = set(int(item) for item in right_row.tolist())
        overlaps.append(len(left_set & right_set) / max(1, len(left_set | right_set)))
    return float(np.mean(overlaps)) if overlaps else 0.0


def _without_arrays(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"top_indices", "top_scores"}}


def _diagnostics_for_indices(
    name: str,
    indices: np.ndarray,
    context_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    predicted: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    if len(indices) == 0:
        return {"scope": name, "available": False, "reason": "empty split"}
    idx = indices.astype("int64")
    target_norm = _normalize(target_embeddings[idx])
    pred_normed = _normalize(predicted[idx])
    pred_cosines = np.sum(pred_normed * target_norm, axis=1)
    rag_current = _retrieval_metrics(context_embeddings[idx], context_embeddings, top_k, true_indices=idx)
    rag_next = _retrieval_metrics(context_embeddings[idx], target_embeddings, top_k, true_indices=idx)
    jepa_next = _retrieval_metrics(predicted[idx], target_embeddings, top_k, true_indices=idx)
    pred_norm = np.linalg.norm(predicted[idx], axis=1)
    return {
        "scope": name,
        "available": True,
        "pred_target_cosine": float(np.mean(pred_cosines)),
        "pred_target_cosine_min": float(np.min(pred_cosines)),
        "pred_target_cosine_max": float(np.max(pred_cosines)),
        "retrieval_hit_at_k": jepa_next["retrieval_hit_at_k"],
        "retrieval_mean_score": jepa_next["retrieval_mean_score"],
        "transition_direction_diversity": jepa_next["transition_direction_diversity"],
        "predicted_vector_norm": float(np.mean(pred_norm)),
        "predicted_vector_norm_min": float(np.min(pred_norm)),
        "predicted_vector_norm_max": float(np.max(pred_norm)),
        "baselines": {
            "rag_current_index": _without_arrays(rag_current),
            "rag_next_index": _without_arrays(rag_next),
            "jepa_next_index": _without_arrays(jepa_next),
        },
        "retrieval_overlap": {
            "jepa_vs_rag_current_index": _overlap(jepa_next["top_indices"], rag_current["top_indices"]),
            "jepa_vs_rag_next_index": _overlap(jepa_next["top_indices"], rag_next["top_indices"]),
            "rag_current_vs_rag_next_index": _overlap(rag_current["top_indices"], rag_next["top_indices"]),
        },
    }


def evaluate_planner_diagnostics(config: AppConfig, top_k: int | None = None) -> dict[str, Any]:
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    if not embeddings_path.exists() or not checkpoint_path.exists():
        return {"available": False, "reason": "missing embeddings or predictor checkpoint"}
    data = np.load(embeddings_path)
    context_embeddings = np.asarray(data["current_embeddings"], dtype="float32")
    target_embeddings = np.asarray(data["next_embeddings"], dtype="float32")
    if len(context_embeddings) == 0:
        return {"available": False, "reason": "empty embeddings"}

    k = top_k or config.generation.top_k
    predicted = _predict_from_embeddings(config, context_embeddings)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_config = checkpoint.get("config", {})
    all_indices = np.arange(len(context_embeddings), dtype="int64")
    val_idx = np.asarray(checkpoint.get("val_idx", []), dtype="int64")
    if len(val_idx) == 0:
        val_idx = all_indices
    val_idx = val_idx[(val_idx >= 0) & (val_idx < len(context_embeddings))]
    all_metrics = _diagnostics_for_indices("all", all_indices, context_embeddings, target_embeddings, predicted, k)
    validation_metrics = _diagnostics_for_indices("validation", val_idx, context_embeddings, target_embeddings, predicted, k)
    primary = validation_metrics if validation_metrics.get("available") else all_metrics
    ablations = {
        "rag_current_index": primary.get("baselines", {}).get("rag_current_index", {}),
        "rag_next_index": primary.get("baselines", {}).get("rag_next_index", {}),
        "jepa_next_index": primary.get("baselines", {}).get("jepa_next_index", {}),
        "jepa_delta_predictor": {
            "available": bool(training_config.get("predict_delta", True)),
            "note": "uses the current checkpoint when predict_delta=true",
            **(primary.get("baselines", {}).get("jepa_next_index", {}) if training_config.get("predict_delta", True) else {}),
        },
        "jepa_no_context_dropout": {
            "available": not bool(training_config.get("use_context_dropout", True)),
            "note": "train a checkpoint with use_context_dropout=false to compare this mode",
        },
    }
    selected_modes = set(config.evaluation.planner_ablation_modes)
    if "rag_current_embedding" in selected_modes:
        selected_modes.add("rag_next_index")
    if "jepa_predicted_target" in selected_modes:
        selected_modes.add("jepa_next_index")
    ablations = {key: value for key, value in ablations.items() if key in selected_modes}
    return {
        "available": True,
        "top_k": int(k),
        "pred_target_cosine": primary.get("pred_target_cosine", 0.0),
        "retrieval_hit_at_k": primary.get("retrieval_hit_at_k", 0.0),
        "retrieval_mean_score": primary.get("retrieval_mean_score", 0.0),
        "transition_direction_diversity": primary.get("transition_direction_diversity", 0.0),
        "predicted_vector_norm": primary.get("predicted_vector_norm", 0.0),
        "validation_pred_target_cosine": validation_metrics.get("pred_target_cosine", 0.0),
        "validation_retrieval_hit_at_k": validation_metrics.get("retrieval_hit_at_k", 0.0),
        "validation_retrieval_mean_score": validation_metrics.get("retrieval_mean_score", 0.0),
        "validation_transition_direction_diversity": validation_metrics.get("transition_direction_diversity", 0.0),
        "all_pred_target_cosine": all_metrics.get("pred_target_cosine", 0.0),
        "all_retrieval_hit_at_k": all_metrics.get("retrieval_hit_at_k", 0.0),
        "all_retrieval_mean_score": all_metrics.get("retrieval_mean_score", 0.0),
        "all_transition_direction_diversity": all_metrics.get("transition_direction_diversity", 0.0),
        "all_predicted_vector_norm": all_metrics.get("predicted_vector_norm", 0.0),
        "validation": validation_metrics,
        "all": all_metrics,
        "retrieval_overlap": primary.get("retrieval_overlap", {}),
        "ablation_modes": ablations,
    }
