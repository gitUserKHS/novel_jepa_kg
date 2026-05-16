from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    chat_model: str = "gemma4:e4b"
    embed_model: str = "embeddinggemma:latest"
    timeout_sec: int = 120
    num_ctx: int = 4096
    num_gpu: int = 40
    num_batch: int = 128
    keep_alive: str = "30s"
    manage_vram: bool = True
    retry_attempts: int = 1
    retry_backoff_sec: float = 2.0
    fallback_num_ctx: int = 3072
    fallback_num_gpu: int = 35
    fallback_num_batch: int = 64
    fallback_max_tokens: int = 1200
    fallback_keep_alive: str = "10s"


class DataConfig(BaseModel):
    synthetic_path: str = "data/synthetic/generated.jsonl"
    filtered_path: str = "data/filtered/filtered.jsonl"
    sample_cache_path: str = "data/synthetic/sample_cache.jsonl"
    embeddings_path: str = "data/embeddings/scenes.npz"
    embedding_cache_path: str = "data/embeddings/embedding_cache.jsonl"
    current_context_index_path: str = "data/indexes/current_context.faiss"
    faiss_index_path: str = "data/indexes/next_scene.faiss"
    min_summary_chars: int = 20
    max_retries: int = 3
    reuse_existing: bool = True
    allow_legacy_sample_cache: bool = True
    diversity_buckets: int = 12
    synthetic_temperature: float = 0.9
    synthetic_max_tokens: int = 1200
    synthetic_candidate_multiplier: float = 1.25


class TrainingConfig(BaseModel):
    model_type: str = "residual_mlp"
    input_window: int = 1
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-4
    val_ratio: float = 0.15
    checkpoint_path: str = "checkpoints/predictor/best.pt"
    hidden_dim: int = 1024
    num_layers: int = 4
    dropout: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 12
    gradient_clip_norm: float = 1.0
    use_amp: bool = False
    use_context_dropout: bool = True
    context_dropout_prob: float = 0.15
    field_dropout_prob: float = 0.20
    normalize_prediction: bool = True
    predict_delta: bool = True
    loss_mse_weight: float = 0.05
    loss_norm_weight: float = 0.001


class GenerationConfig(BaseModel):
    top_k: int = 5
    rag_context_limit: int = 3
    max_tokens: int = 1600
    temperature: float = 0.8
    enable_consistency_repair: bool = True
    use_scene_analyzer: bool = True
    style: str = "한국어 웹소설 문체. 감정선은 선명하게, 장면 전환은 자연스럽게."


class EvaluationConfig(BaseModel):
    use_llm_judge: bool = False
    repetition_ngram: int = 4
    report_dir: str = "reports/runs"
    target_min_chars: int = 600
    target_max_chars: int = 1600
    planner_ablation_modes: list[str] = Field(
        default_factory=lambda: [
            "rag_current_index",
            "rag_next_index",
            "jepa_next_index",
            "jepa_delta_predictor",
            "jepa_no_context_dropout",
        ]
    )


class ChatConfig(BaseModel):
    session_dir: str = "data/sessions"
    recent_messages: int = 8
    compress_every_messages: int = 6
    compress_over_chars: int = 12000
    max_memory_chars: int = 5000
    auto_update_graph: bool = True
    scene_summary_chars: int = 700


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
    chat: ChatConfig = Field(default_factory=ChatConfig)
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
