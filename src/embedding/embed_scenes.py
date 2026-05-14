from __future__ import annotations

import json

import numpy as np

from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def embed_dataset(config: AppConfig, client: OllamaClient) -> dict[str, int]:
    input_path = resolve_path(config, config.data.filtered_path)
    output_path = resolve_path(config, config.data.embeddings_path)
    ensure_parent(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {input_path}")

    samples = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise ValueError("Filtered dataset is empty. Generate and filter data first.")

    current_texts = [sample["scene_t"]["summary"] for sample in samples]
    next_texts = [sample["scene_t_plus_1"]["summary"] for sample in samples]
    logger.info("Embedding %s scene pairs", len(samples))
    current_embeddings = client.embed(current_texts)
    next_embeddings = client.embed(next_texts)
    sample_ids = np.asarray([sample.get("id", idx) for idx, sample in enumerate(samples)], dtype="int64")
    np.savez_compressed(
        output_path,
        current_embeddings=current_embeddings.astype("float32"),
        next_embeddings=next_embeddings.astype("float32"),
        sample_ids=sample_ids,
    )
    return {"count": len(samples), "dim": int(current_embeddings.shape[1])}
