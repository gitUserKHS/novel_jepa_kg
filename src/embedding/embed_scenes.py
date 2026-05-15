from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.llm.ollama_client import OllamaClient
from src.planner.jepa_dataset import build_context_target_texts
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def _embedding_backend(client: OllamaClient) -> str:
    return "dry-run" if client.dry_run else "ollama"


def _text_key(model: str, text: str, backend: str) -> str:
    payload = json.dumps({"backend": backend, "model": model, "text": text}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = row.get("key")
            vector = row.get("vector")
            if key and isinstance(vector, list):
                cache[key] = vector
        except json.JSONDecodeError:
            continue
    return cache


def _write_embedding_cache(path: Path, cache: dict[str, list[float]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for key, vector in cache.items():
            f.write(json.dumps({"key": key, "vector": vector}, ensure_ascii=False) + "\n")


def _texts_hash(model: str, texts: list[str], backend: str) -> np.ndarray:
    return np.asarray([_text_key(model, text, backend) for text in texts], dtype="<U64")


def _can_reuse_embeddings(
    path: Path,
    model: str,
    backend: str,
    current_texts: list[str],
    next_texts: list[str],
    needs_dropout_embeddings: bool,
    dropout_texts: list[str] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        if "current_text_hashes" not in data or "next_text_hashes" not in data or "embed_model" not in data:
            return False
        if "embedding_backend" not in data or str(data["embedding_backend"]) != backend:
            return False
        if "embedding_schema" not in data or str(data["embedding_schema"]) != "jepa-structured-context-v1":
            return False
        if needs_dropout_embeddings:
            if "dropout_context_embeddings" not in data or "dropout_context_hashes" not in data or dropout_texts is None:
                return False
            if not np.array_equal(data["dropout_context_hashes"], _texts_hash(model, dropout_texts, backend)):
                return False
        saved_model = str(data["embed_model"])
        return (
            saved_model == model
            and np.array_equal(data["current_text_hashes"], _texts_hash(model, current_texts, backend))
            and np.array_equal(data["next_text_hashes"], _texts_hash(model, next_texts, backend))
        )
    except Exception:
        return False


def _embed_with_cache(config: AppConfig, client: OllamaClient, texts: list[str]) -> tuple[np.ndarray, int, int]:
    cache_path = resolve_path(config, config.data.embedding_cache_path)
    cache = _load_embedding_cache(cache_path) if config.data.reuse_existing else {}
    backend = _embedding_backend(client)
    keys = [_text_key(config.ollama.embed_model, text, backend) for text in texts]
    missing_texts: list[str] = []
    missing_keys: list[str] = []
    for key, text in zip(keys, texts):
        if key not in cache:
            missing_keys.append(key)
            missing_texts.append(text)

    if missing_texts:
        vectors = client.embed(missing_texts)
        for key, vector in zip(missing_keys, vectors):
            cache[key] = vector.astype("float32").tolist()
        if config.data.reuse_existing:
            _write_embedding_cache(cache_path, cache)

    embeddings = np.asarray([cache[key] for key in keys], dtype="float32")
    return embeddings, len(keys) - len(missing_keys), len(missing_keys)


def embed_dataset(config: AppConfig, client: OllamaClient) -> dict[str, int | bool]:
    input_path = resolve_path(config, config.data.filtered_path)
    output_path = resolve_path(config, config.data.embeddings_path)
    ensure_parent(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {input_path}")

    samples = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise ValueError("Filtered dataset is empty. Generate and filter data first.")

    current_texts, next_texts = build_context_target_texts(samples, seed=config.project.seed)
    backend = _embedding_backend(client)
    needs_dropout_embeddings = bool(config.training.use_context_dropout)
    dropout_texts = None
    if needs_dropout_embeddings:
        dropout_texts, _targets = build_context_target_texts(
            samples,
            use_dropout=True,
            seed=config.project.seed,
            context_dropout_prob=config.training.context_dropout_prob,
            field_dropout_prob=config.training.field_dropout_prob,
        )
    if config.data.reuse_existing and _can_reuse_embeddings(
        output_path,
        config.ollama.embed_model,
        backend,
        current_texts,
        next_texts,
        needs_dropout_embeddings,
        dropout_texts=dropout_texts,
    ):
        data = np.load(output_path)
        return {"count": len(samples), "dim": int(data["current_embeddings"].shape[1]), "reused_file": True}

    logger.info("Embedding %s scene pairs", len(samples))
    current_embeddings, current_reused, current_new = _embed_with_cache(config, client, current_texts)
    next_embeddings, next_reused, next_new = _embed_with_cache(config, client, next_texts)
    dropout_embeddings = None
    dropout_reused = 0
    dropout_new = 0
    if needs_dropout_embeddings and dropout_texts is not None:
        dropout_embeddings, dropout_reused, dropout_new = _embed_with_cache(config, client, dropout_texts)
    sample_ids = np.asarray([sample.get("id", idx) for idx, sample in enumerate(samples)], dtype="int64")
    payload = {
        "current_embeddings": current_embeddings.astype("float32"),
        "next_embeddings": next_embeddings.astype("float32"),
        "sample_ids": sample_ids,
        "current_text_hashes": _texts_hash(config.ollama.embed_model, current_texts, backend),
        "next_text_hashes": _texts_hash(config.ollama.embed_model, next_texts, backend),
        "embed_model": np.asarray(config.ollama.embed_model),
        "embedding_backend": np.asarray(backend),
        "embedding_schema": np.asarray("jepa-structured-context-v1"),
    }
    if dropout_embeddings is not None:
        payload["dropout_context_embeddings"] = dropout_embeddings.astype("float32")
        payload["dropout_context_hashes"] = _texts_hash(config.ollama.embed_model, dropout_texts or [], backend)
    np.savez_compressed(output_path, **payload)
    return {
        "count": len(samples),
        "dim": int(current_embeddings.shape[1]),
        "reused_vectors": current_reused + next_reused + dropout_reused,
        "new_vectors": current_new + next_new + dropout_new,
        "dropout_context_vectors": int(len(dropout_embeddings)) if dropout_embeddings is not None else 0,
        "reused_file": False,
    }
