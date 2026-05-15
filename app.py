from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.diversity import diversity_report_from_samples, training_scale_recommendations
from src.data.filter_dataset import filter_jsonl
from src.data.generate_synthetic import generate_synthetic_dataset
from src.embedding.embed_scenes import embed_dataset
from src.embedding.vector_store import build_current_context_index, build_next_scene_index
from src.evaluation.report import evaluate_and_write_report
from src.generation.chat import CHAT_MODES, generate_chat_turn
from src.generation.generate_baseline import generate_llm_only
from src.generation.generate_with_jepa import generate_with_jepa
from src.generation.generate_with_rag import generate_with_rag, plan_rag_generation
from src.llm.scene_presets import AUTO_SCENE_PRESET, demo_defaults_for_genre, resolve_scene_preset, scene_preset_labels
from src.llm.ollama_client import OllamaClient
from src.memory.context import compress_session_memory, extract_knowledge_graph, graph_tables, graph_to_mermaid
from src.planner.train import train_predictor
from src.planner.predict import evaluate_planner_diagnostics
from src.session.store import (
    create_session,
    delete_session,
    export_session_markdown,
    list_sessions,
    load_session,
    save_session,
)
from src.utils.config import AppConfig, load_config
from src.utils.paths import ensure_project_dirs, resolve_path


st.set_page_config(page_title="Novel JEPA Lab", layout="wide")

PIPELINE_STAGES = [
    {"stage": "Dataset", "work": "Generate or reuse samples, then validate/filter JSONL"},
    {"stage": "Embedding", "work": "Reuse or create scene embeddings, then prepare vectors"},
    {"stage": "Index", "work": "Reuse or build FAISS current-context and next-scene indexes"},
    {"stage": "Train", "work": "Train residual predictor and save best checkpoint"},
    {"stage": "Generate", "work": "Create LLM-only, RAG, and JEPA outputs"},
    {"stage": "Evaluate", "work": "Score outputs and write Markdown report"},
]

DEFAULT_CHAT_SETTINGS = {
    "session_dir": "data/sessions",
    "recent_messages": 8,
    "compress_every_messages": 6,
    "compress_over_chars": 12000,
    "max_memory_chars": 5000,
    "auto_update_graph": True,
    "scene_summary_chars": 700,
}

GENRE_PRESETS = [
    "한국형 SF 미스터리",
    "한국형 판타지 미스터리",
    "궁중 판타지",
    "현대 오컬트",
    "로맨스 스릴러",
    "무협 정치극",
    "디스토피아 성장물",
    "해양 모험",
    "법정 미스터리",
    "학원 이능 배틀",
    "가족 드라마 미스터리",
    "사이버펑크 누아르",
    "역사 대체물",
]

CUSTOM_OPTION = "직접 입력"


def show_error(message: str, exc: Exception | None = None) -> None:
    if exc:
        st.error(f"{message}: {exc}")
    else:
        st.error(message)


def ensure_chat_config(config: AppConfig) -> AppConfig:
    if hasattr(config, "chat"):
        return config
    object.__setattr__(config, "chat", SimpleNamespace(**DEFAULT_CHAT_SETTINGS))
    return config


@st.cache_data(ttl=15, show_spinner=False)
def available_ollama_models(base_url: str, timeout_sec: int) -> list[str]:
    return OllamaClient(base_url=base_url, chat_model="", embed_model="", timeout_sec=timeout_sec).list_models()


@st.cache_data(ttl=5, show_spinner=False)
def running_ollama_models(base_url: str, timeout_sec: int) -> list[dict[str, Any]]:
    return OllamaClient(base_url=base_url, chat_model="", embed_model="", timeout_sec=timeout_sec).running_models()


def model_selector(label: str, current: str, models: list[str], key: str) -> str:
    if not models:
        return st.sidebar.text_input(label, current, key=key)

    options = models + [CUSTOM_OPTION]
    if current in models:
        index = models.index(current)
    elif f"{current}:latest" in models:
        index = models.index(f"{current}:latest")
    else:
        index = len(options) - 1
    selected = st.sidebar.selectbox(label, options, index=index, key=f"{key}_select")
    if selected == CUSTOM_OPTION:
        custom = st.sidebar.text_input(f"Custom {label.lower()}", current, key=key)
        return custom.strip() or current
    return selected


def genre_selector(label: str, default: str, key: str) -> str:
    options = GENRE_PRESETS + [CUSTOM_OPTION]
    index = GENRE_PRESETS.index(default) if default in GENRE_PRESETS else len(options) - 1
    selected = st.selectbox(label, options, index=index, key=f"{key}_preset")
    if selected == CUSTOM_OPTION:
        custom = st.text_input("Custom genre", default if default not in GENRE_PRESETS else "", key=f"{key}_custom")
        return custom.strip() or default
    return selected


def scene_preset_selector(label: str, genre: str, key: str) -> str:
    options = [AUTO_SCENE_PRESET] + scene_preset_labels(genre)
    state_key = f"{key}_scene_preset"
    genre_key = f"{state_key}_genre"
    if st.session_state.get(genre_key) != genre:
        st.session_state[state_key] = AUTO_SCENE_PRESET
        st.session_state[genre_key] = genre
    current = st.session_state.get(state_key)
    if current is not None and current not in options:
        st.session_state[state_key] = AUTO_SCENE_PRESET
    return st.selectbox(label, options, index=0, key=state_key)


def sync_genre_text_defaults(
    key_prefix: str,
    genre: str,
    field_keys: dict[str, str],
    source_values: dict[str, str] | None = None,
) -> None:
    defaults = demo_defaults_for_genre(genre)
    genre_state_key = f"{key_prefix}_defaults_genre"
    previous_genre = st.session_state.get(genre_state_key)
    genre_changed = previous_genre is not None and previous_genre != genre
    for field, state_key in field_keys.items():
        if genre_changed:
            st.session_state[state_key] = defaults[field]
        elif state_key not in st.session_state:
            initial_values = source_values or defaults
            st.session_state[state_key] = initial_values.get(field, defaults[field])
    st.session_state[genre_state_key] = genre


@st.cache_data(show_spinner=False)
def read_jsonl(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_samples(samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(samples):
        plan = sample.get("metadata", {}).get("diversity_plan") or {}
        rows.append(
            {
                "id": sample.get("id", idx),
                "genre": sample.get("world", {}).get("genre", ""),
                "preset": plan.get("label", sample.get("metadata", {}).get("scene_preset_label", "")),
                "pacing": plan.get("pacing", ""),
                "clue": plan.get("clue_type", ""),
                "transition": plan.get("transition_shape", ""),
                "current": sample.get("scene_t", {}).get("summary", ""),
                "next": sample.get("scene_t_plus_1", {}).get("summary", ""),
                "emotion": sample.get("scene_t_plus_1", {}).get("emotion", ""),
            }
        )
    return pd.DataFrame(rows)


def render_diversity_report(report: dict[str, Any]) -> None:
    if not report or not report.get("sample_count"):
        st.caption("No diversity report yet.")
        return
    cols = st.columns(3)
    cols[0].metric("samples", report.get("sample_count", 0))
    cols[1].metric("unique signatures", report.get("unique_signatures", 0))
    cols[2].metric("signature ratio", f"{report.get('signature_ratio', 0.0):.2f}")
    rows = []
    for axis, detail in (report.get("axes") or {}).items():
        rows.append(
            {
                "axis": axis,
                "unique": detail.get("unique", 0),
                "coverage": f"{detail.get('coverage_ratio', 0.0):.2f}",
                "top": " / ".join(
                    f"{item.get('value')} ({item.get('count')})" for item in detail.get("top_values", [])[:3]
                ),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def format_size(path: Path) -> str:
    if not path.exists():
        return "-"
    size = path_total_size(path)
    return format_bytes(size)


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def path_total_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cache_inventory(config: AppConfig) -> list[dict[str, Any]]:
    items = [
        ("Synthetic JSONL", config.data.synthetic_path, "generated data"),
        ("Filtered JSONL", config.data.filtered_path, "generated data"),
        ("Sample cache", config.data.sample_cache_path, "cache"),
        ("Embeddings NPZ", config.data.embeddings_path, "cache"),
        ("Embedding cache", config.data.embedding_cache_path, "cache"),
        ("Current context FAISS", config.data.current_context_index_path, "index"),
        ("Next scene FAISS", config.data.faiss_index_path, "index"),
        ("Predictor checkpoint", config.training.checkpoint_path, "model artifact"),
        ("Model card", "checkpoints/predictor/model_card.json", "model artifact"),
        ("Chat sessions", config.chat.session_dir, "sessions"),
    ]
    rows = []
    for label, relative_path, kind in items:
        path = resolve_path(config, relative_path)
        size = path_total_size(path)
        rows.append(
            {
                "label": label,
                "kind": kind,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": size,
                "size": format_bytes(size),
                "modified": path.stat().st_mtime if path.exists() else 0,
            }
        )
    return rows


def sample_cache_preview(config: AppConfig, limit: int = 5000) -> list[dict[str, Any]]:
    cache_path = resolve_path(config, config.data.sample_cache_path)
    if not cache_path.exists():
        return []
    rows = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = sample.get("metadata", {}) or {}
            plan = metadata.get("diversity_plan") or {}
            rows.append(
                {
                    "line": line_no,
                    "id": sample.get("id", ""),
                    "genre_input": metadata.get("genre_input", ""),
                    "world_genre": sample.get("world", {}).get("genre", ""),
                    "preset": metadata.get("scene_preset_label") or plan.get("label", ""),
                    "plot_function": plan.get("plot_function", ""),
                    "cache_source": metadata.get("cache_source", ""),
                    "key_prefix": str(metadata.get("dataset_key", ""))[:10],
                    "summary": sample.get("scene_t", {}).get("summary", "")[:120],
                }
            )
    return rows


def report_inventory(report_dir: Path) -> list[dict[str, Any]]:
    if not report_dir.exists():
        return []
    files = [path for path in report_dir.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".log", ".txt"}]
    rows = []
    for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "size": format_bytes(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "mtime": stat.st_mtime,
            }
        )
    return rows


def delete_known_paths(paths: list[str]) -> tuple[int, int]:
    deleted = 0
    freed = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        freed += path_total_size(path)
        if path.is_dir():
            for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    item.rmdir()
        else:
            path.unlink(missing_ok=True)
        deleted += 1
    return deleted, freed


def artifact_status(config: AppConfig) -> pd.DataFrame:
    artifacts = [
        ("Synthetic JSONL", config.data.synthetic_path),
        ("Filtered JSONL", config.data.filtered_path),
        ("Sample cache", config.data.sample_cache_path),
        ("Embeddings", config.data.embeddings_path),
        ("Embedding cache", config.data.embedding_cache_path),
        ("Current context index", config.data.current_context_index_path),
        ("Next scene index", config.data.faiss_index_path),
        ("Chat sessions", config.chat.session_dir),
        ("Predictor checkpoint", config.training.checkpoint_path),
        ("Predictor metadata", "checkpoints/predictor/model_card.json"),
        ("Train history", "reports/runs/latest_train_history.json"),
    ]
    rows = []
    for name, relative_path in artifacts:
        path = resolve_path(config, relative_path)
        modified = "-"
        if path.exists():
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        rows.append(
            {
                "artifact": name,
                "status": "ready" if path.exists() else "missing",
                "size": format_size(path),
                "updated": modified,
                "path": str(path),
            }
        )

    report_dir = resolve_path(config, config.evaluation.report_dir)
    reports = sorted(report_dir.glob("comparison_*.md")) if report_dir.exists() else []
    if reports:
        latest = reports[-1]
        rows.append(
            {
                "artifact": "Latest report",
                "status": "ready",
                "size": format_size(latest),
                "updated": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S"),
                "path": str(latest),
            }
        )
    return pd.DataFrame(rows)


def initial_stage_rows() -> list[dict[str, str]]:
    return [
        {"stage": item["stage"], "status": "waiting", "detail": item["work"]}
        for item in PIPELINE_STAGES
    ]


def render_stage_table(placeholder: Any, rows: list[dict[str, str]]) -> None:
    placeholder.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def update_stage(
    rows: list[dict[str, str]],
    placeholder: Any,
    stage_index: int,
    status: str,
    detail: str,
) -> None:
    rows[stage_index]["status"] = status
    rows[stage_index]["detail"] = detail
    render_stage_table(placeholder, rows)


def cache_summary(label: str, data: dict[str, Any]) -> str:
    parts = [label]
    if "generated" in data:
        parts.append(f"new={data.get('generated', 0)}")
    if "reused" in data:
        parts.append(f"reused={data.get('reused', 0)}")
    if "new_vectors" in data:
        parts.append(f"new_vectors={data.get('new_vectors', 0)}")
    if "reused_vectors" in data:
        parts.append(f"cached_vectors={data.get('reused_vectors', 0)}")
    if "dropout_context_vectors" in data:
        parts.append(f"context_dropout={data.get('dropout_context_vectors', 0)}")
    if data.get("reused_file"):
        parts.append("file=reused")
    return " | ".join(parts)


def retrieval_preview_rows(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(retrieved, start=1):
        sample = item.get("sample", {})
        next_scene = sample.get("scene_t_plus_1", {})
        metadata = sample.get("metadata", {})
        rows.append(
            {
                "rank": idx,
                "score": round(float(item.get("score", 0.0)), 4),
                "sample_id": sample.get("id", ""),
                "preset": metadata.get("scene_preset_label", ""),
                "summary": next_scene.get("summary", ""),
                "emotion": next_scene.get("emotion", ""),
                "conflict": next_scene.get("conflict", ""),
            }
        )
    return rows


def make_stream_callback(placeholder: Any) -> Callable[[str], None]:
    buffer: list[str] = []

    def on_chunk(chunk: str) -> None:
        for char in chunk:
            buffer.append(char)
            placeholder.markdown("".join(buffer) + "▌")

    return on_chunk


def make_client(config: AppConfig, dry_run: bool) -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama.base_url,
        chat_model=config.ollama.chat_model,
        embed_model=config.ollama.embed_model,
        timeout_sec=config.ollama.timeout_sec,
        num_ctx=config.ollama.num_ctx,
        num_gpu=config.ollama.num_gpu,
        num_batch=config.ollama.num_batch,
        keep_alive=config.ollama.keep_alive,
        manage_vram=config.ollama.manage_vram,
        dry_run=dry_run,
        retry_attempts=config.ollama.retry_attempts,
        retry_backoff_sec=config.ollama.retry_backoff_sec,
        fallback_num_ctx=config.ollama.fallback_num_ctx,
        fallback_num_gpu=config.ollama.fallback_num_gpu,
        fallback_num_batch=config.ollama.fallback_num_batch,
        fallback_max_tokens=config.ollama.fallback_max_tokens,
        fallback_keep_alive=config.ollama.fallback_keep_alive,
    )


def sidebar_config(config: AppConfig) -> tuple[AppConfig, bool]:
    st.sidebar.header("Project Settings")
    config.ollama.base_url = st.sidebar.text_input("Ollama base URL", config.ollama.base_url)
    try:
        models = available_ollama_models(config.ollama.base_url, config.ollama.timeout_sec)
        st.sidebar.caption(f"Ollama models detected: {len(models)}")
    except Exception as exc:  # noqa: BLE001 - model selection must still work when Ollama is offline.
        models = []
        st.sidebar.warning(f"Could not load Ollama model list: {exc}")
    config.ollama.chat_model = model_selector("Chat model", config.ollama.chat_model, models, "chat_model")
    config.ollama.embed_model = model_selector("Embedding model", config.ollama.embed_model, models, "embed_model")
    config.ollama.num_ctx = int(
        st.sidebar.number_input("Ollama context length", min_value=1024, max_value=32768, value=config.ollama.num_ctx, step=1024)
    )
    config.ollama.num_gpu = int(
        st.sidebar.number_input(
            "Ollama GPU layers",
            min_value=0,
            max_value=99,
            value=config.ollama.num_gpu,
            step=1,
            help="Lower this if gemma4:e4b runner stops. 40 was stable on the target RTX 4060 8GB with ctx 4096 and batch 128.",
        )
    )
    config.ollama.num_batch = int(
        st.sidebar.number_input(
            "Ollama batch size",
            min_value=32,
            max_value=1024,
            value=config.ollama.num_batch,
            step=32,
            help="Lower values reduce VRAM pressure during prompt processing.",
        )
    )
    config.ollama.keep_alive = st.sidebar.text_input("Ollama keep alive", config.ollama.keep_alive)
    config.ollama.manage_vram = st.sidebar.checkbox("Unload other Ollama model before calls", value=config.ollama.manage_vram)
    with st.sidebar.expander("Ollama 500 recovery", expanded=False):
        config.ollama.retry_attempts = int(
            st.number_input(
                "Retry attempts after runner error",
                min_value=0,
                max_value=5,
                value=config.ollama.retry_attempts,
                step=1,
                help="Retries only before any streaming text has been delivered.",
            )
        )
        config.ollama.retry_backoff_sec = float(
            st.number_input(
                "Retry backoff seconds",
                min_value=0.0,
                max_value=30.0,
                value=float(config.ollama.retry_backoff_sec),
                step=0.5,
            )
        )
        config.ollama.fallback_num_ctx = int(
            st.number_input(
                "Fallback context length",
                min_value=1024,
                max_value=32768,
                value=config.ollama.fallback_num_ctx,
                step=1024,
            )
        )
        config.ollama.fallback_num_gpu = int(
            st.number_input(
                "Fallback GPU layers",
                min_value=0,
                max_value=99,
                value=config.ollama.fallback_num_gpu,
                step=1,
            )
        )
        config.ollama.fallback_num_batch = int(
            st.number_input(
                "Fallback batch size",
                min_value=16,
                max_value=1024,
                value=config.ollama.fallback_num_batch,
                step=16,
            )
        )
        config.ollama.fallback_max_tokens = int(
            st.number_input(
                "Fallback max output tokens",
                min_value=256,
                max_value=8192,
                value=config.ollama.fallback_max_tokens,
                step=256,
            )
        )
        config.ollama.fallback_keep_alive = st.text_input("Fallback keep alive", config.ollama.fallback_keep_alive)
    with st.sidebar.expander("Ollama runtime", expanded=False):
        try:
            running = running_ollama_models(config.ollama.base_url, config.ollama.timeout_sec)
            if running:
                rows = []
                for item in running:
                    size = int(item.get("size", 0) or 0)
                    size_vram = int(item.get("size_vram", 0) or 0)
                    gpu_ratio = (size_vram / size * 100) if size else 0.0
                    rows.append(
                        {
                            "model": item.get("name", ""),
                            "gpu": f"{gpu_ratio:.0f}%",
                            "size": f"{size / (1024 ** 3):.1f} GB" if size else "-",
                            "vram": f"{size_vram / (1024 ** 3):.1f} GB" if size_vram else "-",
                            "ctx": item.get("context_length", "-"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("For gemma4:e4b on 8GB VRAM, partial CPU/GPU offload is expected. If runner 500 errors return, lower GPU layers or context length.")
            else:
                st.caption("No Ollama model is currently loaded.")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Runtime status unavailable: {exc}")
    output_root = st.sidebar.text_input("Output directory", ".")
    dry_run = st.sidebar.checkbox("Dry-run mode", value=False)
    if dry_run:
        st.sidebar.warning("Dry-run mode uses deterministic canned outputs. Turn it off for real Ollama prose.")
    config.data.reuse_existing = st.sidebar.checkbox("Reuse cached data", value=config.data.reuse_existing)
    with st.sidebar.expander("Synthetic data controls", expanded=False):
        config.data.diversity_buckets = int(
            st.number_input(
                "Diversity preset buckets",
                min_value=1,
                max_value=64,
                value=config.data.diversity_buckets,
                step=1,
            )
        )
        config.data.synthetic_candidate_multiplier = float(
            st.number_input(
                "Candidate multiplier",
                min_value=1.0,
                max_value=3.0,
                value=float(config.data.synthetic_candidate_multiplier),
                step=0.05,
                help="Allows extra candidate ids when samples fail validation or generation.",
            )
        )
        config.data.allow_legacy_sample_cache = st.checkbox(
            "Allow legacy-compatible sample cache reuse",
            value=config.data.allow_legacy_sample_cache,
            help="Reuse older cached samples when exact v4 diversity cache keys do not match but genre/sample slot/preset are compatible.",
        )
        config.data.synthetic_temperature = float(
            st.number_input(
                "Synthetic temperature",
                min_value=0.1,
                max_value=1.5,
                value=float(config.data.synthetic_temperature),
                step=0.05,
            )
        )
        config.data.synthetic_max_tokens = int(
            st.number_input(
                "Synthetic max tokens",
                min_value=512,
                max_value=4096,
                value=config.data.synthetic_max_tokens,
                step=128,
                help="Lower values speed up JSON sample generation but can increase truncation risk.",
            )
        )
    with st.sidebar.expander("Generation controls", expanded=False):
        config.generation.top_k = int(
            st.number_input("Retrieval top K", min_value=1, max_value=20, value=config.generation.top_k, step=1)
        )
        config.generation.rag_context_limit = int(
            st.number_input(
                "Prompt examples",
                min_value=1,
                max_value=10,
                value=min(config.generation.rag_context_limit, config.generation.top_k),
                step=1,
            )
        )
        config.generation.max_tokens = int(
            st.number_input("Max output tokens", min_value=256, max_value=8192, value=config.generation.max_tokens, step=128)
        )
        config.generation.enable_consistency_repair = st.checkbox(
            "Auto-repair name consistency",
            value=config.generation.enable_consistency_repair,
        )
        config.generation.use_scene_analyzer = st.checkbox(
            "Use current scene analyzer",
            value=config.generation.use_scene_analyzer,
        )
    if output_root.strip() and output_root.strip() != ".":
        config.output_root = output_root.strip()
    return config, dry_run


def run_dataset_stage(
    config: AppConfig,
    client: OllamaClient,
    genre: str,
    count: int,
    scene_preset_label: str | None = None,
) -> dict[str, Any]:
    raw = generate_synthetic_dataset(config, client, genre=genre, count=count, scene_preset=scene_preset_label)
    filtered = filter_jsonl(config)
    read_jsonl.clear()
    filtered_samples = read_jsonl(str(resolve_path(config, config.data.filtered_path)))
    return {"generated": raw, "filtered": filtered, "diversity": diversity_report_from_samples(filtered_samples)}


def run_generation_bundle(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    stream_callbacks: dict[str, Callable[[str], None]] | None = None,
    scene_preset: dict[str, str] | None = None,
) -> dict[str, str]:
    stream_callbacks = stream_callbacks or {}
    return {
        "llm_only": generate_llm_only(
            config,
            client,
            world,
            characters,
            previous_scene,
            stream_callback=stream_callbacks.get("llm_only"),
            scene_preset=scene_preset,
        ),
        "rag": generate_with_rag(
            config,
            client,
            world,
            characters,
            previous_scene,
            stream_callback=stream_callbacks.get("rag"),
            scene_preset=scene_preset,
        ),
        "jepa": generate_with_jepa(
            config,
            client,
            world,
            characters,
            previous_scene,
            stream_callback=stream_callbacks.get("jepa"),
            scene_preset=scene_preset,
        ),
    }


def session_label(summary: dict[str, Any]) -> str:
    title = summary.get("title", "Untitled session")
    updated = summary.get("updated_at", "")
    message_count = summary.get("message_count", 0)
    scene_count = summary.get("scene_count", 0)
    return f"{title} · {message_count} msgs · {scene_count} scenes · {updated}"


def render_chat_session(config: AppConfig, client: OllamaClient) -> None:
    st.subheader("Long-form Chat Session")
    st.caption("장편 소설이 길어질 때 최근 대화, 누적 요약, 지식 그래프를 함께 사용해 컨텍스트를 압축합니다.")

    with st.expander("Create new session", expanded=not list_sessions(config)):
        new_title = st.text_input("Session title", "기억 잔향 연재 세션", key="new_chat_title")
        new_genre = genre_selector("Session genre", "한국형 SF 미스터리", "new_chat_genre")
        sync_genre_text_defaults(
            "new_chat",
            new_genre,
            {
                "world": "new_chat_world",
                "characters": "new_chat_characters",
            },
        )
        new_world = st.text_area("World setting", height=90, key="new_chat_world")
        new_characters = st.text_area(
            "Characters",
            height=90,
            key="new_chat_characters",
        )
        if st.button("Create session", type="primary"):
            session = create_session(config, new_title, new_world, new_characters, genre=new_genre)
            st.session_state["chat_session_id"] = session["session_id"]
            st.rerun()

    sessions = list_sessions(config)
    if not sessions:
        st.info("Create a session to start long-form generation.")
        return

    labels = [session_label(summary) for summary in sessions]
    ids = [summary["session_id"] for summary in sessions]
    selected_id = st.session_state.get("chat_session_id", ids[0])
    selected_index = ids.index(selected_id) if selected_id in ids else 0
    selected_label = st.selectbox("Session", labels, index=selected_index)
    session_id = ids[labels.index(selected_label)]
    st.session_state["chat_session_id"] = session_id
    session = load_session(config, session_id)

    left, right = st.columns([1.35, 1.0])

    with left:
        with st.expander("Session settings", expanded=False):
            session["title"] = st.text_input("Title", session.get("title", ""), key=f"title_{session_id}")
            session_genre = genre_selector(
                "Genre",
                session.get("genre", "한국형 SF 미스터리"),
                f"genre_{session_id}",
            )
            sync_genre_text_defaults(
                f"session_{session_id}",
                session_genre,
                {
                    "world": f"world_{session_id}",
                    "characters": f"characters_{session_id}",
                },
                source_values={
                    "world": session.get("world", ""),
                    "characters": session.get("characters", ""),
                },
            )
            session["genre"] = session_genre
            session["world"] = st.text_area("World", height=100, key=f"world_{session_id}")
            session["characters"] = st.text_area(
                "Characters",
                height=100,
                key=f"characters_{session_id}",
            )
            if st.button("Save settings", key=f"save_settings_{session_id}"):
                save_session(config, session)
                st.success("Session settings saved.")

        st.markdown("#### Chat")
        for message in session.get("messages", []):
            role = message.get("role", "assistant")
            with st.chat_message("user" if role == "user" else "assistant"):
                mode = message.get("mode")
                if mode:
                    st.caption(mode)
                st.markdown(message.get("content", ""))

        mode = st.radio("Generation mode", CHAT_MODES, index=2, horizontal=True, key=f"mode_{session_id}")
        scene_preset_label = scene_preset_selector("Scene preset", session.get("genre", ""), f"chat_{session_id}")
        scene_preset = resolve_scene_preset(session.get("genre", ""), scene_preset_label)
        user_instruction = st.text_area(
            "Next instruction",
            "이전 장면의 감정선을 이어서 다음 장면을 써 주세요. 새 단서와 선택 압박을 포함해 주세요.",
            height=110,
            key=f"instruction_{session_id}",
        )
        if st.button("Generate next scene", type="primary", key=f"generate_{session_id}"):
            try:
                live_output = st.empty()
                with st.status("Generating next scene and updating memory", expanded=True) as status:
                    result = generate_chat_turn(
                        config,
                        client,
                        session,
                        user_instruction,
                        mode,
                        stream_callback=make_stream_callback(live_output),
                        scene_preset=scene_preset,
                    )
                    live_output.markdown(result["assistant_text"])
                    status.update(label="Saved scene, summary, and memory", state="complete")
                st.success(
                    f"Generated {len(result['assistant_text'])} chars. "
                    f"Scene summary: {len(result['scene_summary'])} chars. "
                    f"Compressed: {'yes' if result['compressed'] else 'no'}."
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_error("Chat generation failed", exc)

    with right:
        st.markdown("#### Memory")
        metrics = st.columns(4)
        metrics[0].metric("messages", len(session.get("messages", [])))
        metrics[1].metric("scenes", len(session.get("scene_summaries", [])))
        metrics[2].metric("nodes", len(session.get("knowledge_graph", {}).get("nodes", [])))
        metrics[3].metric("edges", len(session.get("knowledge_graph", {}).get("edges", [])))

        action_cols = st.columns(4)
        if action_cols[0].button("Compress now", key=f"compress_{session_id}"):
            compress_session_memory(config, client, session)
            save_session(config, session)
            st.success("Memory summary updated.")
            st.rerun()
        if action_cols[1].button("Rebuild graph", key=f"graph_{session_id}"):
            all_text = "\n\n".join(
                [session.get("memory_summary", "")]
                + [scene.get("summary", "") for scene in session.get("scene_summaries", [])]
                + [message.get("content", "") for message in session.get("messages", [])[-12:]]
            )
            session["knowledge_graph"] = {"nodes": [], "edges": []}
            extract_knowledge_graph(config, client, session, all_text)
            save_session(config, session)
            st.success("Knowledge graph rebuilt.")
            st.rerun()
        if action_cols[2].button("Export MD", key=f"export_{session_id}"):
            path = export_session_markdown(config, session)
            st.success(f"Exported to {path}")
        if action_cols[3].button("Delete", key=f"delete_{session_id}"):
            delete_session(config, session_id)
            st.session_state.pop("chat_session_id", None)
            st.rerun()

        st.text_area("Memory summary", session.get("memory_summary", ""), height=190, disabled=True)
        scene_rows = [
            {"index": scene.get("index"), "mode": scene.get("mode"), "summary": scene.get("summary")}
            for scene in session.get("scene_summaries", [])
        ]
        st.dataframe(pd.DataFrame(scene_rows), hide_index=True, width="stretch")

        nodes_df, edges_df = graph_tables(session.get("knowledge_graph", {}))
        graph_tabs = st.tabs(["Nodes", "Edges", "Mermaid"])
        with graph_tabs[0]:
            st.dataframe(nodes_df, hide_index=True, width="stretch")
        with graph_tabs[1]:
            st.dataframe(edges_df, hide_index=True, width="stretch")
        with graph_tabs[2]:
            st.code(graph_to_mermaid(session.get("knowledge_graph", {})), language="mermaid")


def main() -> None:
    config = load_config("configs/default.yaml")
    config = ensure_chat_config(config)
    config, dry_run = sidebar_config(config)
    ensure_project_dirs(config)
    client = make_client(config, dry_run)

    st.title("Novel JEPA Lab")
    st.caption("JEPA-inspired latent planner + local LLM Korean novel generation dashboard")

    tabs = st.tabs(["Project", "Chat", "Dataset", "Embedding", "Train", "Generate", "Evaluate", "Reports"])

    with tabs[0]:
        st.subheader("One-click experiment")
        st.write("합성 서사 데이터 생성부터 평가 리포트까지 작은 샘플로 실행합니다.")
        genre = genre_selector("Genre", "한국형 SF 미스터리", "project_genre")
        sync_genre_text_defaults(
            "project",
            genre,
            {
                "world": "project_world",
                "characters": "project_characters",
                "previous_scene": "project_previous_scene",
            },
        )
        scene_preset_label = scene_preset_selector("Scene preset", genre, "project")
        sample_plan = training_scale_recommendations(genre, scene_preset_label, config.data.diversity_buckets)
        plan_cols = st.columns(4)
        plan_cols[0].metric("quick", sample_plan["quick"])
        plan_cols[1].metric("balanced", sample_plan["balanced"])
        plan_cols[2].metric("research", sample_plan["research"])
        plan_cols[3].metric("robust", sample_plan["robust"])
        st.caption(sample_plan["rationale"])
        sample_count = st.number_input("Samples", min_value=2, max_value=500, value=sample_plan["quick"], step=1)
        fresh_dataset = st.checkbox(
            "Create fresh dataset for this run",
            value=False,
            help="Ignore the synthetic sample cache for the full pipeline run and overwrite generated/filtered JSONL.",
        )
        with st.expander("Advanced inputs", expanded=False):
            previous_scene = st.text_area("Previous scene", height=100, key="project_previous_scene")
            world = st.text_area("World setting", height=80, key="project_world")
            characters = st.text_area("Characters", height=80, key="project_characters")
        st.caption("Artifact snapshot")
        st.dataframe(artifact_status(config), hide_index=True, width="stretch")
        if st.button("Run Full Pipeline", type="primary"):
            original_reuse_existing = config.data.reuse_existing
            if fresh_dataset:
                config.data.reuse_existing = False
            progress = st.progress(0)
            stage_rows = initial_stage_rows()
            stage_table = st.empty()
            current_step = st.empty()
            artifact_table = st.empty()
            train_chart = st.empty()
            run_summary: dict[str, Any] = {}
            render_stage_table(stage_table, stage_rows)
            artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
            try:
                dataset_mode = "Generating fresh samples" if fresh_dataset else "Generating or reusing samples"
                update_stage(stage_rows, stage_table, 0, "running", dataset_mode)
                current_step.info("Step 1/6: Dataset generation and validation")
                dataset_result = run_dataset_stage(config, client, genre, int(sample_count), scene_preset_label)
                run_summary["fresh_dataset"] = fresh_dataset
                run_summary["genre"] = genre
                run_summary["scene_preset"] = scene_preset_label
                run_summary["dataset"] = dataset_result
                diversity = dataset_result.get("diversity", {})
                dataset_detail = (
                    f"{cache_summary('samples', dataset_result['generated'])} | "
                    f"legacy={dataset_result['generated'].get('compatible_reused', 0)} | "
                    f"kept={dataset_result['filtered']['kept']} | rejected={dataset_result['filtered']['rejected']} | "
                    f"diversity={diversity.get('unique_signatures', 0)} signatures"
                )
                update_stage(stage_rows, stage_table, 0, "done", dataset_detail)
                with st.expander("Dataset diversity coverage", expanded=False):
                    render_diversity_report(diversity)
                artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
                progress.progress(16)

                update_stage(stage_rows, stage_table, 1, "running", "Embedding only missing vectors")
                current_step.info("Step 2/6: Scene embeddings")
                embed_result = embed_dataset(config, client)
                run_summary["embedding"] = embed_result
                update_stage(stage_rows, stage_table, 1, "done", cache_summary("embeddings", embed_result))
                artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
                progress.progress(32)

                update_stage(stage_rows, stage_table, 2, "running", "Checking FAISS index freshness")
                current_step.info("Step 3/6: Vector indexes")
                current_index_path = build_current_context_index(config)
                next_index_path = build_next_scene_index(config)
                run_summary["index"] = {"current_context": str(current_index_path), "next_scene": str(next_index_path)}
                update_stage(stage_rows, stage_table, 2, "done", f"current={current_index_path} | next={next_index_path}")
                artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
                progress.progress(48)

                train_points: list[dict[str, Any]] = []

                def on_train_epoch(row: dict[str, Any]) -> None:
                    train_points.append(row)
                    epoch = int(row["epoch"])
                    total = int(row["total_epochs"])
                    progress.progress(48 + int(22 * epoch / max(1, total)))
                    update_stage(
                        stage_rows,
                        stage_table,
                        3,
                        "running",
                        f"epoch={epoch}/{total} | val_cosine={row['val_cosine']:.4f} | best={row['best_val_cosine']:.4f}",
                    )
                    train_df = pd.DataFrame(train_points)
                    train_chart.line_chart(train_df.set_index("epoch")[["train_loss", "val_loss", "val_cosine"]])

                update_stage(stage_rows, stage_table, 3, "running", "Training predictor")
                current_step.info("Step 4/6: Predictor training")
                history = train_predictor(config, progress_callback=on_train_epoch)
                run_summary["training"] = {
                    "device": history.get("device"),
                    "parameter_count": history.get("parameter_count"),
                    "best_val_cosine": history.get("best_val_cosine"),
                    "epochs": len(history.get("epochs", [])),
                }
                update_stage(
                    stage_rows,
                    stage_table,
                    3,
                    "done",
                    f"device={history.get('device')} | params={history.get('parameter_count', 0):,} | "
                    f"best_val_cosine={history.get('best_val_cosine', 0):.4f}",
                )
                artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
                progress.progress(70)

                update_stage(stage_rows, stage_table, 4, "running", "Generating comparison outputs")
                current_step.info("Step 5/6: LLM-only, RAG, and JEPA generation")
                generation_views = st.tabs(["LLM only live", "RAG live", "JEPA live"])
                generation_placeholders = {}
                with generation_views[0]:
                    generation_placeholders["llm_only"] = st.empty()
                with generation_views[1]:
                    generation_placeholders["rag"] = st.empty()
                with generation_views[2]:
                    generation_placeholders["jepa"] = st.empty()
                outputs = run_generation_bundle(
                    config,
                    client,
                    world,
                    characters,
                    previous_scene,
                    stream_callbacks={
                        key: make_stream_callback(placeholder)
                        for key, placeholder in generation_placeholders.items()
                    },
                    scene_preset=resolve_scene_preset(genre, scene_preset_label),
                )
                for key, output in outputs.items():
                    generation_placeholders[key].markdown(output or "(empty)")
                run_summary["generation"] = {key: len(value) for key, value in outputs.items()}
                update_stage(stage_rows, stage_table, 4, "done", " / ".join(f"{key}={len(value)} chars" for key, value in outputs.items()))
                progress.progress(88)

                update_stage(stage_rows, stage_table, 5, "running", "Scoring outputs and writing report")
                current_step.info("Step 6/6: Evaluation report")
                report_path = evaluate_and_write_report(
                    config,
                    client,
                    previous_scene,
                    outputs,
                    world=world,
                    characters=characters,
                )
                run_summary["report"] = report_path
                update_stage(stage_rows, stage_table, 5, "done", f"report={report_path}")
                artifact_table.dataframe(artifact_status(config), hide_index=True, width="stretch")
                progress.progress(100)
                current_step.success("Pipeline completed")
                st.success(f"Pipeline completed. Report saved to {report_path}")
                st.json(run_summary)
            except Exception as exc:  # noqa: BLE001 - Streamlit should show readable errors.
                current_step.error("Pipeline failed")
                show_error("Pipeline failed", exc)
            finally:
                config.data.reuse_existing = original_reuse_existing

    with tabs[1]:
        render_chat_session(config, client)

    with tabs[2]:
        st.subheader("Dataset")
        genre = genre_selector("Dataset genre", "한국형 판타지 미스터리", "dataset_genre")
        scene_preset_label = scene_preset_selector("Dataset scene preset", genre, "dataset")
        sample_plan = training_scale_recommendations(genre, scene_preset_label, config.data.diversity_buckets)
        cols = st.columns(4)
        cols[0].metric("quick", sample_plan["quick"])
        cols[1].metric("balanced", sample_plan["balanced"])
        cols[2].metric("research", sample_plan["research"])
        cols[3].metric("robust", sample_plan["robust"])
        st.caption(sample_plan["rationale"])
        count = st.number_input("Number of samples", min_value=1, max_value=1000, value=sample_plan["quick"], step=1)
        if st.button("Generate dataset"):
            try:
                result = run_dataset_stage(config, client, genre, int(count), scene_preset_label)
                generated = result["generated"]
                cols = st.columns(4)
                cols[0].metric("written", generated["written"])
                cols[1].metric("new", generated.get("generated", 0))
                cols[2].metric("reused", generated.get("reused", 0))
                cols[3].metric("kept", result["filtered"]["kept"])
                reuse_cols = st.columns(2)
                reuse_cols[0].metric("exact cache reused", generated.get("exact_reused", 0))
                reuse_cols[1].metric("legacy-compatible reused", generated.get("compatible_reused", 0))
                st.caption(
                    f"Checked {generated.get('candidates_checked', generated.get('written', 0))} "
                    f"of {generated.get('candidate_limit', generated.get('requested', 0))} candidate ids."
                )
                render_diversity_report(result.get("diversity", {}))
                st.success(
                    "Generated "
                    f"{generated['written']} samples "
                    f"({generated.get('generated', 0)} new, {generated.get('reused', 0)} reused), "
                    f"kept {result['filtered']['kept']}."
                )
            except Exception as exc:  # noqa: BLE001
                show_error("Dataset generation failed", exc)
        samples = read_jsonl(str(resolve_path(config, config.data.filtered_path)))
        with st.expander("Current filtered dataset diversity", expanded=False):
            render_diversity_report(diversity_report_from_samples(samples))
        st.dataframe(flatten_samples(samples), width="stretch")

    with tabs[3]:
        st.subheader("Embedding")
        if st.button("Embed filtered dataset"):
            try:
                result = embed_dataset(config, client)
                current_index_path = build_current_context_index(config)
                index_path = build_next_scene_index(config)
                cols = st.columns(4)
                cols[0].metric("pairs", result["count"])
                cols[1].metric("new vectors", result.get("new_vectors", 0))
                cols[2].metric("cached vectors", result.get("reused_vectors", 0))
                cols[3].metric("file reused", "yes" if result.get("reused_file") else "no")
                cache_note = (
                    "reused existing embedding file"
                    if result.get("reused_file")
                    else f"{result.get('new_vectors', 0)} new vectors, {result.get('reused_vectors', 0)} cached vectors"
                )
                st.success(
                    f"Saved {result['count']} embedding pairs ({cache_note}) and FAISS indexes to "
                    f"{current_index_path} and {index_path}."
                )
            except Exception as exc:  # noqa: BLE001
                show_error("Embedding failed", exc)
        st.code(str(resolve_path(config, config.data.embeddings_path)))

    with tabs[4]:
        st.subheader("Train")
        try:
            import torch

            if torch.cuda.is_available():
                device_label = torch.cuda.get_device_name(0)
                st.info(f"Training device: {device_label} | torch {torch.__version__} | CUDA {torch.version.cuda}")
            else:
                st.info(f"Training device: CPU | torch {torch.__version__}")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not inspect torch device: {exc}")
        st.info("Planner type: JEPA-inspired latent transition planner with frozen text encoders and a trainable PyTorch predictor.")
        st.caption(f"Scene analyzer for inference: {'on' if config.generation.use_scene_analyzer else 'off'}")
        filtered_samples = read_jsonl(str(resolve_path(config, config.data.filtered_path)))
        train_cols = st.columns(3)
        train_cols[0].metric("filtered samples", len(filtered_samples))
        train_cols[1].metric("recommended minimum", 32)
        train_cols[2].metric("research target", 96)
        if len(filtered_samples) < 32:
            st.warning("For JEPA diagnostics, generate more samples first. 8-24 is fine for smoke tests, but 32+ is a better minimum.")
        with st.expander("Training data diversity", expanded=False):
            render_diversity_report(diversity_report_from_samples(filtered_samples))
        config.training.epochs = int(st.number_input("Epochs", min_value=1, max_value=500, value=config.training.epochs))
        config.training.batch_size = int(
            st.number_input("Batch size", min_value=1, max_value=512, value=config.training.batch_size)
        )
        config.training.learning_rate = float(
            st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=config.training.learning_rate, format="%0.6f")
        )
        col_a, col_b = st.columns(2)
        with col_a:
            config.training.hidden_dim = int(
                st.number_input("Hidden dim", min_value=128, max_value=4096, value=config.training.hidden_dim, step=128)
            )
            config.training.num_layers = int(
                st.number_input("Layers", min_value=2, max_value=12, value=config.training.num_layers, step=1)
            )
            config.training.dropout = float(
                st.number_input("Dropout", min_value=0.0, max_value=0.8, value=config.training.dropout, step=0.05)
            )
        with col_b:
            config.training.weight_decay = float(
                st.number_input("Weight decay", min_value=0.0, max_value=0.2, value=config.training.weight_decay, format="%0.4f")
            )
            config.training.early_stopping_patience = int(
                st.number_input(
                    "Early stop patience",
                    min_value=0,
                    max_value=100,
                    value=config.training.early_stopping_patience,
                )
            )
            config.training.use_amp = st.checkbox("Use AMP on CUDA", value=config.training.use_amp)
            config.training.predict_delta = st.checkbox("Predict delta", value=config.training.predict_delta)
            config.training.normalize_prediction = st.checkbox("Normalize prediction", value=config.training.normalize_prediction)
        with st.expander("JEPA-inspired planner options", expanded=False):
            config.training.use_context_dropout = st.checkbox(
                "Use context dropout",
                value=config.training.use_context_dropout,
            )
            config.training.context_dropout_prob = float(
                st.number_input(
                    "Context dropout probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.training.context_dropout_prob,
                    step=0.05,
                )
            )
            config.training.field_dropout_prob = float(
                st.number_input(
                    "Field dropout probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.training.field_dropout_prob,
                    step=0.05,
                )
            )
            config.training.loss_mse_weight = float(
                st.number_input("Loss MSE weight", min_value=0.0, max_value=1.0, value=config.training.loss_mse_weight, format="%0.4f")
            )
            config.training.loss_norm_weight = float(
                st.number_input("Loss norm weight", min_value=0.0, max_value=0.1, value=config.training.loss_norm_weight, format="%0.4f")
            )
        model_card_path = resolve_path(config, "checkpoints/predictor/model_card.json")
        if model_card_path.exists():
            with st.expander("Latest model card", expanded=False):
                model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
                summary_cols = st.columns(3)
                summary_cols[0].metric("validation pred_target_cosine", f"{model_card.get('best_pred_target_cosine', 0.0):.4f}")
                summary_cols[1].metric("effective norm weight", f"{model_card.get('effective_loss_norm_weight', 0.0):.4f}")
                summary_cols[2].metric("params", f"{model_card.get('parameter_count', 0):,}")
                st.json(model_card)
        try:
            diagnostics = evaluate_planner_diagnostics(config, top_k=config.generation.top_k)
            if diagnostics.get("available"):
                metric_cols = st.columns(3)
                metric_cols[0].metric("validation retrieval hit@k", f"{diagnostics.get('validation_retrieval_hit_at_k', 0.0):.4f}")
                metric_cols[1].metric("validation mean score", f"{diagnostics.get('validation_retrieval_mean_score', 0.0):.4f}")
                metric_cols[2].metric("validation diversity", f"{diagnostics.get('validation_transition_direction_diversity', 0.0):.4f}")
        except Exception:
            pass
        if st.button("Train predictor"):
            try:
                train_progress = st.progress(0)
                train_status = st.empty()
                live_chart = st.empty()
                train_points: list[dict[str, Any]] = []

                def on_train_epoch(row: dict[str, Any]) -> None:
                    train_points.append(row)
                    epoch = int(row["epoch"])
                    total = int(row["total_epochs"])
                    train_progress.progress(int(100 * epoch / max(1, total)))
                    train_status.info(
                        f"epoch={epoch}/{total} | pred_target_cosine={row.get('val_pred_target_cosine', row['val_cosine']):.4f} | "
                        f"best={row['best_val_cosine']:.4f}"
                    )
                    train_df = pd.DataFrame(train_points)
                    chart_cols = [
                        col
                        for col in ["train_loss", "val_loss", "val_pred_target_cosine", "val_predicted_vector_norm"]
                        if col in train_df.columns
                    ]
                    live_chart.line_chart(train_df.set_index("epoch")[chart_cols])

                history = train_predictor(config, progress_callback=on_train_epoch)
                train_status.success("Training completed")
                st.success(
                    f"Best checkpoint saved to {resolve_path(config, config.training.checkpoint_path)} "
                    f"({history.get('parameter_count', 0):,} params, device={history.get('device')})"
                )
                history_df = pd.DataFrame(history["epochs"])
                chart_cols = [
                    col
                    for col in ["train_loss", "val_loss", "val_pred_target_cosine", "val_predicted_vector_norm"]
                    if col in history_df.columns
                ]
                st.plotly_chart(
                    px.line(history_df, x="epoch", y=chart_cols),
                    width="stretch",
                )
            except Exception as exc:  # noqa: BLE001
                show_error("Training failed", exc)

    with tabs[5]:
        st.subheader("Generate")
        generation_genre = genre_selector("Generation genre", "한국형 SF 미스터리", "generation_genre")
        sync_genre_text_defaults(
            "generation",
            generation_genre,
            {
                "world": "gen_world",
                "characters": "gen_chars",
                "previous_scene": "gen_prev",
            },
        )
        scene_preset_label = scene_preset_selector("Scene preset", generation_genre, "generation")
        scene_preset = resolve_scene_preset(generation_genre, scene_preset_label)
        world = st.text_area("World", height=80, key="gen_world")
        characters = st.text_area("Characters", height=80, key="gen_chars")
        previous_scene = st.text_area(
            "Previous scene summary",
            height=100,
            key="gen_prev",
        )
        mode = st.radio("Mode", ["LLM only", "RAG + LLM", "JEPA Planner + RAG + LLM"], horizontal=True)
        if st.button("Generate prose"):
            try:
                live_output = st.empty()
                stream_callback = make_stream_callback(live_output)
                generation_details: dict[str, Any] = {}
                if mode == "LLM only":
                    output = generate_llm_only(
                        config,
                        client,
                        world,
                        characters,
                        previous_scene,
                        stream_callback=stream_callback,
                        scene_preset=scene_preset,
                    )
                elif mode == "RAG + LLM":
                    result = generate_with_rag(
                        config,
                        client,
                        world,
                        characters,
                        previous_scene,
                        stream_callback=stream_callback,
                        scene_preset=scene_preset,
                        return_details=True,
                    )
                    output = str(result["text"])
                    generation_details = result.get("rag", {})
                else:
                    result = generate_with_jepa(
                        config,
                        client,
                        world,
                        characters,
                        previous_scene,
                        stream_callback=stream_callback,
                        scene_preset=scene_preset,
                        return_details=True,
                    )
                    output = str(result["text"])
                    generation_details = result.get("planner", {})
                    generation_details["rag_baselines"] = plan_rag_generation(
                        config,
                        client,
                        world,
                        characters,
                        previous_scene,
                        scene_preset=scene_preset,
                    )
                live_output.markdown(output)
                if generation_details:
                    st.markdown("#### Retrieval / Planner diagnostics")
                    analyzed_scene = generation_details.get("analyzed_scene")
                    if analyzed_scene:
                        st.markdown("##### Analyzed current scene")
                        st.json(analyzed_scene)
                    if generation_details.get("direction"):
                        st.info(f"Predicted direction: {generation_details['direction']}")
                    if "predicted_vector_norm" in generation_details:
                        st.metric("predicted vector norm", f"{generation_details['predicted_vector_norm']:.4f}")
                    retrieved = generation_details.get("retrieved", [])
                    if retrieved and mode == "JEPA Planner + RAG + LLM":
                        st.markdown("##### JEPA retrieved examples")
                        st.dataframe(pd.DataFrame(retrieval_preview_rows(retrieved)), hide_index=True, width="stretch")
                    if generation_details.get("current_retrieved"):
                        st.markdown("##### RAG current-index retrieved examples")
                        st.dataframe(
                            pd.DataFrame(retrieval_preview_rows(generation_details["current_retrieved"])),
                            hide_index=True,
                            width="stretch",
                        )
                    if generation_details.get("next_retrieved"):
                        st.markdown("##### RAG next-index retrieved examples")
                        st.dataframe(
                            pd.DataFrame(retrieval_preview_rows(generation_details["next_retrieved"])),
                            hide_index=True,
                            width="stretch",
                        )
                    rag_baselines = generation_details.get("rag_baselines", {})
                    if rag_baselines.get("current_retrieved"):
                        st.markdown("##### RAG current-index retrieved examples")
                        st.dataframe(
                            pd.DataFrame(retrieval_preview_rows(rag_baselines["current_retrieved"])),
                            hide_index=True,
                            width="stretch",
                        )
                    if rag_baselines.get("next_retrieved"):
                        st.markdown("##### RAG next-index retrieved examples")
                        st.dataframe(
                            pd.DataFrame(retrieval_preview_rows(rag_baselines["next_retrieved"])),
                            hide_index=True,
                            width="stretch",
                        )
            except Exception as exc:  # noqa: BLE001
                show_error("Generation failed", exc)

    with tabs[6]:
        st.subheader("Evaluate")
        previous_scene = st.text_area(
            "Reference previous scene",
            "하린은 언니의 기억 조각이 불법 경매에 올라왔다는 사실을 알게 된다.",
            height=80,
            key="eval_prev",
        )
        eval_world = st.text_area("World setting for consistency check", height=80, key="eval_world")
        eval_characters = st.text_area("Known characters for consistency check", height=80, key="eval_characters")
        llm_only = st.text_area("LLM-only output", height=120)
        rag = st.text_area("RAG output", height=120)
        jepa = st.text_area("JEPA output", height=120)
        if st.button("Write evaluation report"):
            try:
                outputs = {"llm_only": llm_only, "rag": rag, "jepa": jepa}
                report_path = evaluate_and_write_report(
                    config,
                    client,
                    previous_scene,
                    outputs,
                    world=eval_world,
                    characters=eval_characters,
                )
                st.success(f"Report saved to {report_path}")
            except Exception as exc:  # noqa: BLE001
                show_error("Evaluation failed", exc)

    with tabs[7]:
        st.subheader("Reports / Storage")
        report_dir = resolve_path(config, config.evaluation.report_dir)
        storage_tabs = st.tabs(["Cache / artifacts", "Report cleanup", "Report viewer"])

        with storage_tabs[0]:
            rows = cache_inventory(config)
            total_size = sum(row["size_bytes"] for row in rows)
            st.metric("tracked storage", format_bytes(total_size))
            budget_mb = st.number_input("Cache budget MB", min_value=1, max_value=100000, value=2048, step=128)
            budget_bytes = int(budget_mb * 1024 * 1024)
            if total_size > budget_bytes:
                st.warning(f"Tracked files exceed budget by {format_bytes(total_size - budget_bytes)}.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "label": row["label"],
                            "kind": row["kind"],
                            "exists": row["exists"],
                            "size": row["size"],
                            "path": row["path"],
                        }
                        for row in rows
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            with st.expander("Sample cache browser", expanded=False):
                cache_rows = sample_cache_preview(config)
                if cache_rows:
                    cache_filter = st.text_input("Filter cache rows by genre/preset/plot", "")
                    filtered_cache_rows = cache_rows
                    if cache_filter.strip():
                        needle = cache_filter.strip().lower()
                        filtered_cache_rows = [
                            row
                            for row in cache_rows
                            if needle
                            in " ".join(
                                str(row.get(key, ""))
                                for key in ["genre_input", "world_genre", "preset", "plot_function", "summary"]
                            ).lower()
                        ]
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("cached samples shown", len(filtered_cache_rows))
                    metric_cols[1].metric("cached samples total", len(cache_rows))
                    metric_cols[2].metric("legacy reuse", "on" if config.data.allow_legacy_sample_cache else "off")
                    st.dataframe(pd.DataFrame(filtered_cache_rows), hide_index=True, width="stretch")
                    st.caption(
                        "Exact reuse requires the current schema/model/genre/sample slot/diversity plan key. "
                        "Legacy-compatible reuse can reuse older rows when genre, sample slot, preset, and plot function line up."
                    )
                else:
                    st.caption("No sample cache rows found.")
            delete_labels = st.multiselect(
                "Cache/artifact files to delete",
                [row["label"] for row in rows if row["exists"]],
                help="Deletes only known project files listed above. Deleted artifacts can be regenerated by the pipeline.",
            )
            confirm_cache_delete = st.checkbox("Confirm cache/artifact deletion")
            if st.button("Delete selected cache/artifact files", type="secondary"):
                if not confirm_cache_delete:
                    st.warning("Check the confirmation box first.")
                else:
                    selected_paths = [row["path"] for row in rows if row["label"] in delete_labels]
                    deleted, freed = delete_known_paths(selected_paths)
                    st.success(f"Deleted {deleted} item(s), freed {format_bytes(freed)}.")
                    st.rerun()

        with storage_tabs[1]:
            report_rows = report_inventory(report_dir)
            report_total = sum(row["size_bytes"] for row in report_rows)
            cols = st.columns(3)
            cols[0].metric("report files", len(report_rows))
            cols[1].metric("report storage", format_bytes(report_total))
            cols[2].metric("report directory", str(report_dir))
            if report_rows:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "name": row["name"],
                                "size": row["size"],
                                "modified": row["modified"],
                            }
                            for row in report_rows
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            keep_latest = st.number_input("Keep latest report/log files", min_value=0, max_value=500, value=10, step=1)
            delete_report_names = st.multiselect(
                "Specific report/log files to delete",
                [row["name"] for row in report_rows],
            )
            confirm_report_delete = st.checkbox("Confirm report cleanup")
            report_actions = st.columns(2)
            if report_actions[0].button("Delete selected reports/logs"):
                if not confirm_report_delete:
                    st.warning("Check the confirmation box first.")
                else:
                    selected_paths = [row["path"] for row in report_rows if row["name"] in delete_report_names]
                    deleted, freed = delete_known_paths(selected_paths)
                    st.success(f"Deleted {deleted} report/log file(s), freed {format_bytes(freed)}.")
                    st.rerun()
            if report_actions[1].button("Keep latest and delete older"):
                if not confirm_report_delete:
                    st.warning("Check the confirmation box first.")
                else:
                    old_paths = [row["path"] for row in sorted(report_rows, key=lambda item: item["mtime"], reverse=True)[int(keep_latest) :]]
                    deleted, freed = delete_known_paths(old_paths)
                    st.success(f"Deleted {deleted} older report/log file(s), freed {format_bytes(freed)}.")
                    st.rerun()

        with storage_tabs[2]:
            reports = sorted(report_dir.glob("*.md")) if report_dir.exists() else []
            selected = st.selectbox("Report", [p.name for p in reports]) if reports else None
            if selected:
                st.markdown((report_dir / selected).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
