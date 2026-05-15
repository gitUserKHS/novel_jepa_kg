from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.embedding.vector_store import retrieve_by_vector
from src.llm.ollama_client import OllamaClient
from src.planner.jepa_dataset import build_generation_context_text
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


def predict_next_embedding(
    config: AppConfig,
    client: OllamaClient,
    previous_scene: str,
    world: str = "",
    characters: str = "",
    scene_preset: dict[str, str] | None = None,
) -> np.ndarray:
    context_text = build_generation_context_text(world, characters, previous_scene, scene_preset=scene_preset)
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
    context_text = build_generation_context_text(world, characters, previous_scene, scene_preset=scene_preset)
    context = client.embed([context_text])
    predicted = _predict_from_embeddings(config, context)[0].astype("float32")
    retrieved = retrieve_by_vector(config, predicted, config.generation.top_k)
    return {
        "context_text": context_text,
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


def _retrieval_metrics(query_vectors: np.ndarray, target_vectors: np.ndarray, top_k: int) -> dict[str, Any]:
    query = _normalize(query_vectors)
    target = _normalize(target_vectors)
    similarity = query @ target.T
    indices, scores = _topk(similarity, top_k)
    hits = [int(row_idx in set(row.tolist())) for row_idx, row in enumerate(indices)]
    unique_targets = len({int(idx) for row in indices for idx in row.tolist()})
    return {
        "retrieval_hit_at_k": float(np.mean(hits)) if hits else 0.0,
        "retrieval_mean_score": float(np.mean(scores)) if scores.size else 0.0,
        "transition_direction_diversity": float(unique_targets / max(1, indices.size)),
        "top_indices": indices,
        "top_scores": scores,
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
    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = _normalize(target_embeddings)
    pred_normed = _normalize(predicted)
    pred_cosines = np.sum(pred_normed * target_norm, axis=1)
    jepa_metrics = _retrieval_metrics(predicted, target_embeddings, k)
    rag_metrics = _retrieval_metrics(context_embeddings, target_embeddings, k)
    jepa_top = jepa_metrics.pop("top_indices")
    rag_top = rag_metrics.pop("top_indices")
    jepa_metrics.pop("top_scores", None)
    rag_metrics.pop("top_scores", None)
    overlaps = []
    for left, right in zip(jepa_top, rag_top):
        left_set = set(int(item) for item in left.tolist())
        right_set = set(int(item) for item in right.tolist())
        overlaps.append(len(left_set & right_set) / max(1, len(left_set | right_set)))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_config = checkpoint.get("config", {})
    ablations = {
        "rag_current_embedding": rag_metrics,
        "jepa_predicted_target": jepa_metrics,
        "jepa_delta_predictor": {
            "available": bool(training_config.get("predict_delta", True)),
            "note": "uses the current checkpoint when predict_delta=true",
            **(jepa_metrics if training_config.get("predict_delta", True) else {}),
        },
        "jepa_no_context_dropout": {
            "available": not bool(training_config.get("use_context_dropout", True)),
            "note": "train a checkpoint with use_context_dropout=false to compare this mode",
        },
    }
    selected_modes = set(config.evaluation.planner_ablation_modes)
    ablations = {key: value for key, value in ablations.items() if key in selected_modes}
    return {
        "available": True,
        "top_k": int(k),
        "pred_target_cosine": float(np.mean(pred_cosines)),
        "pred_target_cosine_min": float(np.min(pred_cosines)),
        "pred_target_cosine_max": float(np.max(pred_cosines)),
        "retrieval_hit_at_k": jepa_metrics["retrieval_hit_at_k"],
        "retrieval_mean_score": jepa_metrics["retrieval_mean_score"],
        "transition_direction_diversity": jepa_metrics["transition_direction_diversity"],
        "jepa_vs_rag_retrieval_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "predicted_vector_norm": float(np.mean(pred_norm)),
        "predicted_vector_norm_min": float(np.min(pred_norm)),
        "predicted_vector_norm_max": float(np.max(pred_norm)),
        "ablation_modes": ablations,
    }
