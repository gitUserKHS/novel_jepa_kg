from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype("float32")


def build_next_scene_index(config: AppConfig) -> Path:
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    index_path = resolve_path(config, config.data.faiss_index_path)
    ensure_parent(index_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    data = np.load(embeddings_path)
    next_embeddings = _normalize(data["next_embeddings"])
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required to build the vector index. Install requirements.txt.") from exc
    index = faiss.IndexFlatIP(next_embeddings.shape[1])
    index.add(next_embeddings)
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        faiss.write_index(index, str(tmp_path))
        shutil.move(str(tmp_path), index_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return index_path


def retrieve_by_vector(config: AppConfig, query_vector: np.ndarray, top_k: int) -> list[dict]:
    index_path = resolve_path(config, config.data.faiss_index_path)
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    filtered_path = resolve_path(config, config.data.filtered_path)
    if not index_path.exists():
        build_next_scene_index(config)
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


def retrieve_by_text(config: AppConfig, client: OllamaClient, text: str, top_k: int) -> list[dict]:
    vector = client.embed([text])[0]
    return retrieve_by_vector(config, vector, top_k)
