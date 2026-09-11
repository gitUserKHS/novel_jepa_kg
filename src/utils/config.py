from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


DEFAULT_CHAT_MODEL = "gemma4:e4b"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = "embeddinggemma:latest"
    timeout_sec: int = 300
    # Measured against dry-run long-form turns: the assembled prose prompt grows
    # to ~14.6K characters by turn 6 and converges near 15K, because the recent
    # excerpt, story memory, and consumed-beat budgets are each capped. At
    # roughly 1.5-2 Korean characters per token that is ~8-10K input tokens,
    # and `generation.max_tokens` adds 1800 more. The old 8192 window could not
    # hold that, and prose_prompt() front-loads [세계관] and [인물], so an
    # overflow drops the story canon first. 16384 covers the measured plateau
    # with margin; raising it further mainly costs KV cache on an 8GB card.
    num_ctx: int = 16384
    num_gpu: int = 24
    num_batch: int = 32
    keep_alive: str = "60s"
    top_p: float = 0.92
    repeat_penalty: float = 1.12
    manage_vram: bool = True
    retry_attempts: int = 1
    retry_backoff_sec: float = 2.0
    # The recovery path trades quality for VRAM, but 4096 truncated a ~10K-token
    # prompt hard enough to guarantee a broken section. Keep the savings in
    # num_gpu/num_batch/max_tokens and leave enough window to hold the prompt.
    fallback_num_ctx: int = 8192
    fallback_num_gpu: int = 16
    fallback_num_batch: int = 16
    fallback_max_tokens: int = 1200
    fallback_keep_alive: str = "10s"


DEFAULT_NOVEL_MODEL = "hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M"


class LLMConfig(BaseModel):
    """소설 생성 백엔드. 기본은 Ollama 의 Gemma4 26B-A4B (MoE, 활성 4B, Q4_K_M, 17GB)."""

    backend: str = "ollama"  # "ollama" | "local" (레거시 Qwen3.5-4B 모델 서버)
    model: str = DEFAULT_NOVEL_MODEL  # Ollama 모델 태그 (ollama pull 로 받은 그대로)
    ollama_base_url: str = "http://127.0.0.1:11434"
    # 09-03 실측 (RTX 4060 8GB): 한국어 1.5자/토큰, 장 프롬프트 ~15K자 ≈ 10K 토큰 + 생성 1,500 토큰.
    # 16384 면 여유가 있고 KV 포함 VRAM 4.6 GiB. 요청마다 명시해야 하며(Ollama 기본 4096 은 작품 설정·인물을
    # 소리 없이 잘라 낸다), 값을 바꾸면 모델을 다시 올리므로 서비스 안에서는 고정한다.
    num_ctx: int = 16384
    keep_alive: str = "30m"  # 턴 사이에 모델을 내리지 않는다
    base_url: str = "http://127.0.0.1:8765"  # local 백엔드(모델 서버) 주소
    dry_run: bool = False  # 테스트/CI: 모델 없이 고정 응답
    timeout_sec: int = 1800
    korean_filter: bool = True  # local 백엔드 전용 — Ollama 에는 토큰 필터가 없다
    novel_temperature: float = 0.75
    novel_top_p: float = 0.9
    novel_top_k: int = 40
    novel_repetition_penalty: float = 1.05
    novel_max_tokens: int = 1500
    # DRY(구절 반복 억제) — local 백엔드 전용, 0 이면 꺼짐. min_p 는 0 이면 top_p 사용. 값은 bench_prose_settings 로 결정.
    novel_dry_multiplier: float = 0.8
    novel_dry_base: float = 1.75
    novel_dry_allowed_length: int = 2
    novel_min_p: float = 0.0
    memory_max_tokens: int = 600  # 장 요약 메모리 JSON
    plan_max_tokens: int = 400  # 장 설계 JSON
    review_max_tokens: int = 600  # 개연성 검토 JSON


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
    history_path: str = "reports/runs/latest_train_history.json"
    model_card_path: str = "checkpoints/predictor/model_card.json"
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
    representation_regularizer: str = "target_visreg"
    regularization_weight: float = 0.03
    regularization_min_samples: int = 40
    regularization_min_batch_size: int = 8
    regularization_num_slices: int = 64
    regularization_variance_floor_ratio: float = 0.75
    regularization_center_weight: float = 0.1


class GenerationConfig(BaseModel):
    top_k: int = 5
    rag_context_limit: int = 3
    max_tokens: int = 1800
    temperature: float = 0.8
    sectioned_output: bool = True
    section_count: int = 15
    section_min_chars: int = 1800
    target_novel_chars: int = 30000
    turn_target_chars: int = 5000
    turn_max_sections: int = 8
    longform_max_sections: int = 60
    # 직전 장 꼬리 원문. 앞 장들은 요약·상태 원장으로만 넣는다 (직전 장 전문은 복사 루프의 원인, 08-23 실측).
    longform_recent_context_chars: int = 2600
    longform_checkpoint_path: str = "reports/runs/creative_longform_latest.md"
    longform_state_path: str = "reports/runs/creative_longform_state.json"
    enable_story_memory_rag: bool = True
    story_memory_top_k: int = 4
    story_memory_context_chars: int = 3200
    story_memory_path: str = "reports/runs/creative_longform_memory.jsonl"
    story_ledger_path: str = "reports/runs/creative_longform_ledger.json"
    story_outline_path: str = "reports/runs/creative_longform_outline.json"
    story_summary_group_size: int = 4
    enable_consumed_beat_ledger: bool = True
    enable_repetition_retry: bool = True
    repetition_retry_temperature_delta: float = -0.25
    consumed_beat_context_chars: int = 1200
    hallucination_target: float = 0.35
    hallucination_temperature_delta: float = 0.15
    hallucination_temperature_span: float = 0.8
    enable_consistency_repair: bool = False
    enable_story_outline: bool = True
    outline_beat_count: int = 12
    # 개연성 장치 (2026-09-03, src/generation/plausibility.py): 장을 쓰기 전에 인과 설계를 받고, 쓴 뒤 연속성
    # 편집자 역할의 모델이 확정 사실·앞선 사건·직전 장면과의 모순과 원인 없는 사건을 1~10 으로 검토한다.
    # 모순이 하나라도 있거나 점수가 기준 미만이면 문제 목록을 들고 한 번 고쳐 쓴 뒤 다시 검토한다.
    enable_scene_plan: bool = True
    enable_plausibility_gate: bool = True
    plausibility_min_score: int = 7
    enable_stability_retry: bool = True
    stability_min_section_ratio: float = 0.55
    enable_jepa_coherence_gate: bool = True
    # Calibrated 2026-08-06 against artifact 20260806T111529Z-7262bfe2 (112
    # samples) with embeddinggemma, over two genres: 6 genuine sections scored
    # 0.651-0.766 and 6 deliberate causal breaks scored 0.502-0.637. At 0.64 no
    # genuine section is rewritten and every break is caught. Genre shifts the
    # scale (fantasy floor 0.732, courtroom floor 0.651), so recalibrate with
    # scripts/calibrate_coherence.py after retraining the predictor, changing
    # the embedding model, or writing in a very different genre.
    jepa_coherence_min_cosine: float = 0.64
    jepa_coherence_keep_chat_loaded: bool = True
    use_scene_analyzer: bool = True
    genre: str = ""
    style: str = "한국어 웹소설 문체. 감정선은 선명하게, 장면 전환은 자연스럽게."


class EvaluationConfig(BaseModel):
    use_llm_judge: bool = False
    # The judge reads a bounded head/tail excerpt so a 30K-char draft still
    # fits the 8K-token local context alongside the scene and setting text.
    judge_temperature: float = 0.2
    judge_max_tokens: int = 700
    judge_excerpt_chars: int = 4000
    repetition_ngram: int = 4
    report_dir: str = "reports/runs"
    target_min_chars: int = 27000
    target_max_chars: int = 36000
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


class ServiceConfig(BaseModel):
    name: str = "Novel JEPA Lab Admin"
    bind_host: str = "127.0.0.1"
    port: int = 8502
    require_access_token: bool = False
    access_token_env: str = "NOVEL_JEPA_ACCESS_TOKEN"
    job_lock_path: str = ".runtime/service.job.lock"


class ConsumerConfig(BaseModel):
    name: str = "이야기 공방"
    bind_host: str = "0.0.0.0"
    port: int = 8501
    auth_session_days: int = 7
    database_path: str = ".runtime/consumer.sqlite3"
    story_root: str = "data/consumer_stories"
    retention_days: int = 30
    worker_poll_sec: float = 2.0
    # One worker is enough, and each extra one reserves gigabytes of commit
    # charge that llama-server then cannot allocate. An OS-level lock keeps a
    # leaked launcher from silently stacking hidden workers.
    worker_lock_path: str = ".runtime/consumer.worker.lock"
    worker_heartbeat_sec: float = 5.0
    stale_job_sec: int = 90
    # 자동 이어쓰기: 사람이 다음 전개를 적지 않아도 워커가 연달아 넣는 턴의 상한 (폭주 방지).
    auto_continue_max_turns: int = 20
    allowed_turn_chars: list[int] = Field(default_factory=lambda: [2000, 3000, 5000])
    default_turn_chars: int = 3000
    min_target_chars: int = 10000
    default_target_chars: int = 30000
    max_target_chars: int = 50000
    target_char_step: int = 1000
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = "embeddinggemma:latest"
    active_manifest_path: str = "artifacts/active.json"
    candidates_dir: str = "artifacts/candidates"
    versions_dir: str = "artifacts/versions"
    min_samples: int = 40
    recommended_samples: int = 96
    min_validation_samples: int = 6
    min_validation_cosine: float = 0.60
    min_hit_at_5: float = 0.50
    min_top1_diversity: float = 0.40
    min_effective_rank_ratio: float = 0.50
    min_jepa_gain_over_rag_next: float = 0.03

    def target_char_options(self) -> list[int]:
        minimum = max(self.default_turn_chars, self.min_target_chars)
        maximum = max(minimum, self.max_target_chars)
        step = max(1, self.target_char_step)
        values = list(range(minimum, maximum + 1, step))
        values.extend([self.default_target_chars, maximum])
        return sorted({value for value in values if minimum <= value <= maximum})


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    consumer: ConsumerConfig = Field(default_factory=ConsumerConfig)
    output_root: str = "."


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value!r}")


def apply_environment_overrides(config: AppConfig) -> AppConfig:
    config.ollama.base_url = _first_env("NOVEL_JEPA_OLLAMA_BASE_URL", "OLLAMA_BASE_URL") or config.ollama.base_url
    config.ollama.chat_model = _first_env("NOVEL_JEPA_CHAT_MODEL", "OLLAMA_CHAT_MODEL") or config.ollama.chat_model
    config.ollama.embed_model = _first_env("NOVEL_JEPA_EMBED_MODEL", "OLLAMA_EMBED_MODEL") or config.ollama.embed_model
    config.output_root = _first_env("NOVEL_JEPA_OUTPUT_ROOT") or config.output_root
    config.llm.backend = _first_env("NOVEL_LLM_BACKEND") or config.llm.backend
    config.llm.model = _first_env("NOVEL_LLM_MODEL") or config.llm.model
    config.llm.ollama_base_url = (
        _first_env("NOVEL_LLM_OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "NOVEL_JEPA_OLLAMA_BASE_URL")
        or config.llm.ollama_base_url
    )
    config.llm.base_url = _first_env("NOVEL_LLM_BASE_URL", "NOVEL_QWEN_BASE_URL") or config.llm.base_url
    config.llm.dry_run = _env_bool(_first_env("NOVEL_LLM_DRY_RUN"), config.llm.dry_run)
    config.service.name = _first_env("NOVEL_JEPA_SERVICE_NAME") or config.service.name
    config.service.bind_host = _first_env("NOVEL_JEPA_BIND_HOST") or config.service.bind_host
    port = _first_env("NOVEL_JEPA_PORT")
    if port is not None:
        config.service.port = int(port)
    config.service.require_access_token = _env_bool(
        _first_env("NOVEL_JEPA_REQUIRE_AUTH"),
        config.service.require_access_token,
    )
    config.consumer.bind_host = (
        _first_env("NOVEL_JEPA_CONSUMER_BIND_HOST") or config.consumer.bind_host
    )
    consumer_port = _first_env("NOVEL_JEPA_CONSUMER_PORT")
    if consumer_port is not None:
        config.consumer.port = int(consumer_port)
    config.consumer.database_path = (
        _first_env("NOVEL_JEPA_CONSUMER_DB") or config.consumer.database_path
    )
    config.consumer.story_root = (
        _first_env("NOVEL_JEPA_CONSUMER_STORY_ROOT") or config.consumer.story_root
    )
    return config


def load_config(path: str | Path) -> AppConfig:
    load_dotenv(override=False)
    config_path = Path(path)
    if not config_path.exists():
        return apply_environment_overrides(AppConfig())
    try:
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc
    return apply_environment_overrides(AppConfig(**raw))
