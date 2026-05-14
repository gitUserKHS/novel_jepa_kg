from __future__ import annotations

import json
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
from src.generation.generate_baseline import generate_llm_only
from src.generation.generate_with_jepa import generate_with_jepa
from src.generation.generate_with_rag import generate_with_rag
from src.llm.ollama_client import OllamaClient
from src.planner.train import train_predictor
from src.utils.config import AppConfig, load_config
from src.utils.paths import ensure_project_dirs, resolve_path


st.set_page_config(page_title="Novel JEPA Lab", layout="wide")


def show_error(message: str, exc: Exception | None = None) -> None:
    if exc:
        st.error(f"{message}: {exc}")
    else:
        st.error(message)


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


def main() -> None:
    config = load_config("configs/default.yaml")
    config, dry_run = sidebar_config(config)
    ensure_project_dirs(config)
    client = make_client(config, dry_run)

    st.title("Novel JEPA Lab")
    st.caption("JEPA-inspired latent planner + local LLM Korean novel generation dashboard")

    tabs = st.tabs(["Project", "Dataset", "Embedding", "Train", "Generate", "Evaluate", "Reports"])

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
        if st.button("Run Full Pipeline", type="primary"):
            progress = st.progress(0)
            try:
                with st.status("Generating and filtering dataset"):
                    run_dataset_stage(config, client, genre, int(sample_count))
                progress.progress(20)
                with st.status("Embedding scenes"):
                    embed_dataset(config, client)
                    build_next_scene_index(config)
                progress.progress(45)
                with st.status("Training predictor"):
                    history = train_predictor(config)
                progress.progress(70)
                with st.status("Generating comparison outputs"):
                    outputs = run_generation_bundle(config, client, world, characters, previous_scene)
                progress.progress(90)
                with st.status("Writing evaluation report"):
                    report_path = evaluate_and_write_report(config, client, previous_scene, outputs)
                progress.progress(100)
                st.success(f"Pipeline completed. Report saved to {report_path}")
                st.json({"history": history, "outputs": outputs})
            except Exception as exc:  # noqa: BLE001 - Streamlit should show readable errors.
                show_error("Pipeline failed", exc)

    with tabs[1]:
        st.subheader("Dataset")
        genre = st.text_input("Dataset genre", "한국형 판타지 미스터리")
        count = st.number_input("Number of samples", min_value=1, max_value=500, value=10, step=1)
        if st.button("Generate dataset"):
            try:
                result = run_dataset_stage(config, client, genre, int(count))
                generated = result["generated"]
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

    with tabs[2]:
        st.subheader("Embedding")
        if st.button("Embed filtered dataset"):
            try:
                result = embed_dataset(config, client)
                index_path = build_next_scene_index(config)
                cache_note = (
                    "reused existing embedding file"
                    if result.get("reused_file")
                    else f"{result.get('new_vectors', 0)} new vectors, {result.get('reused_vectors', 0)} cached vectors"
                )
                st.success(f"Saved {result['count']} embedding pairs ({cache_note}) and FAISS index to {index_path}.")
            except Exception as exc:  # noqa: BLE001
                show_error("Embedding failed", exc)
        st.code(str(resolve_path(config, config.data.embeddings_path)))

    with tabs[3]:
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
                history = train_predictor(config)
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

    with tabs[4]:
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

    with tabs[5]:
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

    with tabs[6]:
        st.subheader("Reports")
        report_dir = resolve_path(config, config.evaluation.report_dir)
        reports = sorted(report_dir.glob("*.md")) if report_dir.exists() else []
        selected = st.selectbox("Report", [p.name for p in reports]) if reports else None
        if selected:
            st.markdown((report_dir / selected).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
