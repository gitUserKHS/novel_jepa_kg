from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype("float32")


def _build_index(config: AppConfig, index_path: Path, vector_key: str) -> Path:
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    ensure_parent(index_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if config.data.reuse_existing and index_path.exists() and index_path.stat().st_mtime >= embeddings_path.stat().st_mtime:
        return index_path
    data = np.load(embeddings_path)
    embeddings = _normalize(data[vector_key])
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required to build the vector index. Install requirements.txt.") from exc
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        faiss.write_index(index, str(tmp_path))
        shutil.move(str(tmp_path), index_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return index_path


def build_current_context_index(config: AppConfig) -> Path:
    return _build_index(config, resolve_path(config, config.data.current_context_index_path), "current_embeddings")


def build_next_scene_index(config: AppConfig) -> Path:
    return _build_index(config, resolve_path(config, config.data.faiss_index_path), "next_embeddings")


def _load_index_results(config: AppConfig, query_vector: np.ndarray, top_k: int, index_path: Path, build_fn: Any) -> list[dict]:
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    filtered_path = resolve_path(config, config.data.filtered_path)
    if not index_path.exists():
        build_fn(config)
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for retrieval. Install requirements.txt.") from exc
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copyfile(index_path, tmp_path)
        index = faiss.read_index(str(tmp_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    query = _normalize(query)
    scores, indices = index.search(query, top_k)
    samples = [json.loads(line) for line in filtered_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= int(idx) < len(samples):
            sample = samples[int(idx)]
            result.append({"score": float(score), "sample": sample})
    return result


def retrieve_current_context_by_vector(config: AppConfig, query_vector: np.ndarray, top_k: int) -> list[dict]:
    return _load_index_results(
        config,
        query_vector,
        top_k,
        resolve_path(config, config.data.current_context_index_path),
        build_current_context_index,
    )


def retrieve_current_context_by_text(config: AppConfig, client: OllamaClient, text: str, top_k: int) -> list[dict]:
    vector = client.embed([text])[0]
    return retrieve_current_context_by_vector(config, vector, top_k)


def retrieve_next_by_vector(config: AppConfig, query_vector: np.ndarray, top_k: int) -> list[dict]:
    return _load_index_results(
        config,
        query_vector,
        top_k,
        resolve_path(config, config.data.faiss_index_path),
        build_next_scene_index,
    )


def retrieve_next_by_text(config: AppConfig, client: OllamaClient, text: str, top_k: int) -> list[dict]:
    vector = client.embed([text])[0]
    return retrieve_next_by_vector(config, vector, top_k)


def retrieve_by_vector(config: AppConfig, query_vector: np.ndarray, top_k: int) -> list[dict]:
    return retrieve_next_by_vector(config, query_vector, top_k)


def retrieve_by_text(config: AppConfig, client: OllamaClient, text: str, top_k: int) -> list[dict]:
    return retrieve_next_by_text(config, client, text, top_k)
