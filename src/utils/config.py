from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    chat_model: str = "gemma4:e4b"
    embed_model: str = "embeddinggemma"
    timeout_sec: int = 120


class DataConfig(BaseModel):
    synthetic_path: str = "data/synthetic/generated.jsonl"
    filtered_path: str = "data/filtered/filtered.jsonl"
    sample_cache_path: str = "data/synthetic/sample_cache.jsonl"
    embeddings_path: str = "data/embeddings/scenes.npz"
    embedding_cache_path: str = "data/embeddings/embedding_cache.jsonl"
    faiss_index_path: str = "data/indexes/next_scene.faiss"
    min_summary_chars: int = 20
    max_retries: int = 3
    reuse_existing: bool = True
    diversity_buckets: int = 12


class TrainingConfig(BaseModel):
    model_type: str = "residual_mlp"
    input_window: int = 1
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    val_ratio: float = 0.15
    checkpoint_path: str = "checkpoints/predictor/best.pt"
    hidden_dim: int = 1024
    num_layers: int = 4
    dropout: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
    use_amp: bool = False


class GenerationConfig(BaseModel):
    top_k: int = 5
    max_tokens: int = 1600
    temperature: float = 0.8
    style: str = "한국어 웹소설 문체. 감정선은 선명하게, 장면 전환은 자연스럽게."


class EvaluationConfig(BaseModel):
    use_llm_judge: bool = False
    repetition_ngram: int = 4
    report_dir: str = "reports/runs"
    target_min_chars: int = 600
    target_max_chars: int = 1600


class ProjectConfig(BaseModel):
    name: str = "Novel JEPA Lab"
    language: str = "ko"
    seed: int = 42


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_root: str = "."


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    try:
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc
    return AppConfig(**raw)
