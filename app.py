from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.filter_dataset import filter_jsonl
from src.data.generate_synthetic import generate_synthetic_dataset
from src.embedding.embed_scenes import embed_dataset
from src.embedding.vector_store import build_next_scene_index
from src.evaluation.report import evaluate_and_write_report
from src.generation.chat import CHAT_MODES, generate_chat_turn
from src.generation.generate_baseline import generate_llm_only
from src.generation.generate_with_jepa import generate_with_jepa
from src.generation.generate_with_rag import generate_with_rag
from src.llm.ollama_client import OllamaClient
from src.memory.context import compress_session_memory, extract_knowledge_graph, graph_tables, graph_to_mermaid
from src.planner.train import train_predictor
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
    {"stage": "Index", "work": "Reuse or build FAISS next-scene index"},
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
        rows.append(
            {
                "id": sample.get("id", idx),
                "genre": sample.get("world", {}).get("genre", ""),
                "current": sample.get("scene_t", {}).get("summary", ""),
                "next": sample.get("scene_t_plus_1", {}).get("summary", ""),
                "emotion": sample.get("scene_t_plus_1", {}).get("emotion", ""),
            }
        )
    return pd.DataFrame(rows)


def format_size(path: Path) -> str:
    if not path.exists():
        return "-"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def artifact_status(config: AppConfig) -> pd.DataFrame:
    artifacts = [
        ("Synthetic JSONL", config.data.synthetic_path),
        ("Filtered JSONL", config.data.filtered_path),
        ("Sample cache", config.data.sample_cache_path),
        ("Embeddings", config.data.embeddings_path),
        ("Embedding cache", config.data.embedding_cache_path),
        ("FAISS index", config.data.faiss_index_path),
        ("Chat sessions", config.chat.session_dir),
        ("Predictor checkpoint", config.training.checkpoint_path),
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
    placeholder.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


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
    if data.get("reused_file"):
        parts.append("file=reused")
    return " | ".join(parts)


def make_client(config: AppConfig, dry_run: bool) -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama.base_url,
        chat_model=config.ollama.chat_model,
        embed_model=config.ollama.embed_model,
        timeout_sec=config.ollama.timeout_sec,
        dry_run=dry_run,
    )


def sidebar_config(config: AppConfig) -> tuple[AppConfig, bool]:
    st.sidebar.header("Project Settings")
    config.ollama.base_url = st.sidebar.text_input("Ollama base URL", config.ollama.base_url)
    config.ollama.chat_model = st.sidebar.text_input("Chat model", config.ollama.chat_model)
    config.ollama.embed_model = st.sidebar.text_input("Embedding model", config.ollama.embed_model)
    output_root = st.sidebar.text_input("Output directory", ".")
    dry_run = st.sidebar.checkbox("Dry-run mode", value=True)
    config.data.reuse_existing = st.sidebar.checkbox("Reuse cached data", value=config.data.reuse_existing)
    if output_root.strip() and output_root.strip() != ".":
        config.output_root = output_root.strip()
    return config, dry_run


def run_dataset_stage(config: AppConfig, client: OllamaClient, genre: str, count: int) -> dict[str, Any]:
    raw = generate_synthetic_dataset(config, client, genre=genre, count=count)
    filtered = filter_jsonl(config)
    read_jsonl.clear()
    return {"generated": raw, "filtered": filtered}


def run_generation_bundle(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
) -> dict[str, str]:
    return {
        "llm_only": generate_llm_only(config, client, world, characters, previous_scene),
        "rag": generate_with_rag(config, client, world, characters, previous_scene),
        "jepa": generate_with_jepa(config, client, world, characters, previous_scene),
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
        new_world = st.text_area("World setting", "기억이 물리적 흔적으로 남는 근미래 서울.", height=90, key="new_chat_world")
        new_characters = st.text_area(
            "Characters",
            "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
            height=90,
            key="new_chat_characters",
        )
        if st.button("Create session", type="primary"):
            session = create_session(config, new_title, new_world, new_characters)
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
            session["world"] = st.text_area("World", session.get("world", ""), height=100, key=f"world_{session_id}")
            session["characters"] = st.text_area(
                "Characters",
                session.get("characters", ""),
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
        user_instruction = st.text_area(
            "Next instruction",
            "이전 장면의 감정선을 이어서 다음 장면을 써 주세요. 새 단서와 선택 압박을 포함해 주세요.",
            height=110,
            key=f"instruction_{session_id}",
        )
        if st.button("Generate next scene", type="primary", key=f"generate_{session_id}"):
            try:
                with st.status("Generating next scene and updating memory", expanded=True) as status:
                    result = generate_chat_turn(config, client, session, user_instruction, mode)
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
        st.dataframe(pd.DataFrame(scene_rows), hide_index=True, use_container_width=True)

        nodes_df, edges_df = graph_tables(session.get("knowledge_graph", {}))
        graph_tabs = st.tabs(["Nodes", "Edges", "Mermaid"])
        with graph_tabs[0]:
            st.dataframe(nodes_df, hide_index=True, use_container_width=True)
        with graph_tabs[1]:
            st.dataframe(edges_df, hide_index=True, use_container_width=True)
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
        genre = st.text_input("Genre", "한국형 SF 미스터리", key="project_genre")
        sample_count = st.number_input("Samples", min_value=2, max_value=100, value=8, step=1)
        previous_scene = st.text_area(
            "Previous scene",
            "주인공은 폐쇄된 연구동에서 사라진 동생의 이름이 적힌 실험 기록을 발견한다.",
            height=100,
        )
        world = st.text_area("World setting", "기억이 물리적 흔적으로 남는 근미래 서울.", height=80)
        characters = st.text_area("Characters", "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.", height=80)
        st.caption("Artifact snapshot")
        st.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
        if st.button("Run Full Pipeline", type="primary"):
            progress = st.progress(0)
            stage_rows = initial_stage_rows()
            stage_table = st.empty()
            current_step = st.empty()
            artifact_table = st.empty()
            train_chart = st.empty()
            run_summary: dict[str, Any] = {}
            render_stage_table(stage_table, stage_rows)
            artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
            try:
                update_stage(stage_rows, stage_table, 0, "running", "Generating or reusing samples")
                current_step.info("Step 1/6: Dataset generation and validation")
                dataset_result = run_dataset_stage(config, client, genre, int(sample_count))
                run_summary["dataset"] = dataset_result
                dataset_detail = (
                    f"{cache_summary('samples', dataset_result['generated'])} | "
                    f"kept={dataset_result['filtered']['kept']} | rejected={dataset_result['filtered']['rejected']}"
                )
                update_stage(stage_rows, stage_table, 0, "done", dataset_detail)
                artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
                progress.progress(16)

                update_stage(stage_rows, stage_table, 1, "running", "Embedding only missing vectors")
                current_step.info("Step 2/6: Scene embeddings")
                embed_result = embed_dataset(config, client)
                run_summary["embedding"] = embed_result
                update_stage(stage_rows, stage_table, 1, "done", cache_summary("embeddings", embed_result))
                artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
                progress.progress(32)

                update_stage(stage_rows, stage_table, 2, "running", "Checking FAISS index freshness")
                current_step.info("Step 3/6: Vector index")
                index_path = build_next_scene_index(config)
                run_summary["index"] = str(index_path)
                update_stage(stage_rows, stage_table, 2, "done", f"index={index_path}")
                artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
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
                artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
                progress.progress(70)

                update_stage(stage_rows, stage_table, 4, "running", "Generating comparison outputs")
                current_step.info("Step 5/6: LLM-only, RAG, and JEPA generation")
                outputs = run_generation_bundle(config, client, world, characters, previous_scene)
                run_summary["generation"] = {key: len(value) for key, value in outputs.items()}
                update_stage(stage_rows, stage_table, 4, "done", " / ".join(f"{key}={len(value)} chars" for key, value in outputs.items()))
                progress.progress(88)

                update_stage(stage_rows, stage_table, 5, "running", "Scoring outputs and writing report")
                current_step.info("Step 6/6: Evaluation report")
                report_path = evaluate_and_write_report(config, client, previous_scene, outputs)
                run_summary["report"] = report_path
                update_stage(stage_rows, stage_table, 5, "done", f"report={report_path}")
                artifact_table.dataframe(artifact_status(config), hide_index=True, use_container_width=True)
                progress.progress(100)
                current_step.success("Pipeline completed")
                st.success(f"Pipeline completed. Report saved to {report_path}")
                st.json(run_summary)
            except Exception as exc:  # noqa: BLE001 - Streamlit should show readable errors.
                current_step.error("Pipeline failed")
                show_error("Pipeline failed", exc)

    with tabs[1]:
        render_chat_session(config, client)

    with tabs[2]:
        st.subheader("Dataset")
        genre = st.text_input("Dataset genre", "한국형 판타지 미스터리")
        count = st.number_input("Number of samples", min_value=1, max_value=500, value=10, step=1)
        if st.button("Generate dataset"):
            try:
                result = run_dataset_stage(config, client, genre, int(count))
                generated = result["generated"]
                cols = st.columns(4)
                cols[0].metric("written", generated["written"])
                cols[1].metric("new", generated.get("generated", 0))
                cols[2].metric("reused", generated.get("reused", 0))
                cols[3].metric("kept", result["filtered"]["kept"])
                st.success(
                    "Generated "
                    f"{generated['written']} samples "
                    f"({generated.get('generated', 0)} new, {generated.get('reused', 0)} reused), "
                    f"kept {result['filtered']['kept']}."
                )
            except Exception as exc:  # noqa: BLE001
                show_error("Dataset generation failed", exc)
        samples = read_jsonl(str(resolve_path(config, config.data.filtered_path)))
        st.dataframe(flatten_samples(samples), use_container_width=True)

    with tabs[3]:
        st.subheader("Embedding")
        if st.button("Embed filtered dataset"):
            try:
                result = embed_dataset(config, client)
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
                st.success(f"Saved {result['count']} embedding pairs ({cache_note}) and FAISS index to {index_path}.")
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
                        f"epoch={epoch}/{total} | val_cosine={row['val_cosine']:.4f} | "
                        f"best={row['best_val_cosine']:.4f}"
                    )
                    train_df = pd.DataFrame(train_points)
                    live_chart.line_chart(train_df.set_index("epoch")[["train_loss", "val_loss", "val_cosine"]])

                history = train_predictor(config, progress_callback=on_train_epoch)
                train_status.success("Training completed")
                st.success(
                    f"Best checkpoint saved to {resolve_path(config, config.training.checkpoint_path)} "
                    f"({history.get('parameter_count', 0):,} params, device={history.get('device')})"
                )
                history_df = pd.DataFrame(history["epochs"])
                st.plotly_chart(
                    px.line(history_df, x="epoch", y=["train_loss", "val_loss", "val_cosine"]),
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                show_error("Training failed", exc)

    with tabs[5]:
        st.subheader("Generate")
        world = st.text_area("World", "기억 조각을 거래하는 근미래 도시.", height=80, key="gen_world")
        characters = st.text_area("Characters", "하린: 실종된 언니를 찾는 복원사. 도겸: 기억 암시장의 브로커.", height=80, key="gen_chars")
        previous_scene = st.text_area(
            "Previous scene summary",
            "하린은 언니의 기억 조각이 불법 경매에 올라왔다는 사실을 알게 된다.",
            height=100,
            key="gen_prev",
        )
        mode = st.radio("Mode", ["LLM only", "RAG + LLM", "JEPA Planner + RAG + LLM"], horizontal=True)
        if st.button("Generate prose"):
            try:
                if mode == "LLM only":
                    output = generate_llm_only(config, client, world, characters, previous_scene)
                elif mode == "RAG + LLM":
                    output = generate_with_rag(config, client, world, characters, previous_scene)
                else:
                    output = generate_with_jepa(config, client, world, characters, previous_scene)
                st.markdown(output)
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
        llm_only = st.text_area("LLM-only output", height=120)
        rag = st.text_area("RAG output", height=120)
        jepa = st.text_area("JEPA output", height=120)
        if st.button("Write evaluation report"):
            try:
                outputs = {"llm_only": llm_only, "rag": rag, "jepa": jepa}
                report_path = evaluate_and_write_report(config, client, previous_scene, outputs)
                st.success(f"Report saved to {report_path}")
            except Exception as exc:  # noqa: BLE001
                show_error("Evaluation failed", exc)

    with tabs[7]:
        st.subheader("Reports")
        report_dir = resolve_path(config, config.evaluation.report_dir)
        reports = sorted(report_dir.glob("*.md")) if report_dir.exists() else []
        selected = st.selectbox("Report", [p.name for p in reports]) if reports else None
        if selected:
            st.markdown((report_dir / selected).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
