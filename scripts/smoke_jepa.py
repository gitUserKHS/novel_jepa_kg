from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filter_dataset import filter_jsonl
from src.data.diversity import diversity_report_from_samples, training_scale_recommendations
from src.data.generate_synthetic import generate_synthetic_dataset
from src.embedding.embed_scenes import embed_dataset
from src.embedding.vector_store import build_current_context_index, build_next_scene_index, retrieve_by_vector
from src.generation.generate_with_jepa import generate_with_jepa
from src.generation.generate_with_rag import generate_with_rag
from src.generation.hallucination import (
    export_longform_bundle,
    generate_with_controlled_hallucination,
    import_longform_file,
    longform_artifact_status,
)
from src.llm.ollama_client import OllamaClient
from src.memory.story_rag import (
    MEMORY_END,
    MEMORY_START,
    StoryMemory,
    StoryMemoryStreamFilter,
    build_story_ledger,
    format_hierarchical_story_context,
    retrieve_story_memories,
    split_story_memory,
)
from src.planner.jepa_dataset import MASK_TOKEN, build_context_text, build_target_text
from src.planner.jepa_model import JEPAPredictor
from src.planner.jepa_train import representation_prediction_loss, train_predictor
from src.planner.predict import evaluate_planner_diagnostics, predict_next_embedding
from src.planner.scene_analyzer import analyze_current_scene, build_analyzed_generation_context
from src.utils.config import AppConfig
from src.utils.paths import ensure_project_dirs, resolve_path


def _sample() -> dict:
    return {
        "world": {"genre": "한국형 SF 미스터리", "premise": "기억 잔향 도시", "rules": ["기억은 흔적으로 남는다"]},
        "characters": [
            {
                "name": "서윤",
                "role": "기록 복원가",
                "goal": "동생 찾기",
                "fear": "기억 상실",
                "relationship": "민재를 의심한다",
            }
        ],
        "scene_t": {
            "summary": "서윤은 폐쇄 연구동에서 손상된 로그를 발견한다.",
            "emotion": "불안",
            "conflict": "로그를 복원하면 위치가 노출된다.",
            "state": ["단서 발견"],
            "plot_function": "단서 발견",
        },
        "scene_t_plus_1": {
            "summary": "서윤은 로그를 복원하고 숨겨진 실험실 좌표를 얻는다.",
            "emotion": "결심",
            "conflict": "좌표를 따라가면 추적자가 붙는다.",
            "state": ["좌표 확보", "추적 위험"],
            "plot_function": "위기 고조",
        },
        "metadata": {
            "scene_preset_label": "기록 장치 활성화",
            "diversity_plan": {"next_hook": "실험실 안쪽에서 살아 있는 신호가 응답한다"},
        },
    }


def test_builders_and_loss() -> None:
    sample = _sample()
    context = build_context_text(sample)
    target = build_target_text(sample)
    dropped = build_context_text(sample, use_dropout=True, context_dropout_prob=1.0, field_dropout_prob=1.0)
    assert "숨겨진 실험실 좌표를 얻는다" not in context
    assert "다음 장면 요약" in target
    assert MASK_TOKEN in dropped
    scale = training_scale_recommendations("?쒓뎅??SF 誘몄뒪?곕━", None, 5)
    assert scale["quick"] >= 8

    model = JEPAPredictor(dim=384, hidden_dim=128, num_layers=3)
    x = torch.randn(2, 384)
    y = torch.randn(2, 384)
    pred = model(x)
    assert pred.shape == y.shape
    loss, cosine, mse, norm = representation_prediction_loss(pred, y)
    assert loss.item() >= 0
    assert cosine.item() >= 0
    assert mse.item() >= 0
    assert norm.item() >= 0

    config = AppConfig()
    client = OllamaClient(
        config.ollama.base_url,
        config.ollama.chat_model,
        config.ollama.embed_model,
        num_ctx=4096,
        num_gpu=40,
        num_batch=128,
        retry_attempts=1,
        fallback_num_ctx=3072,
        fallback_num_gpu=35,
        fallback_num_batch=64,
        fallback_max_tokens=1200,
        dry_run=True,
    )
    fallback_options = client._chat_options(0.8, 1600, recovery_mode=True)
    assert fallback_options["num_ctx"] == 3072
    assert fallback_options["num_gpu"] == 35
    assert fallback_options["num_batch"] == 64
    assert fallback_options["num_predict"] == 1200
    analysis = analyze_current_scene(
        config,
        client,
        "기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
        "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
        "서윤은 폐쇄 연구동에서 손상된 로그와 동생의 이름을 발견한다.",
    )
    for key in ["summary", "emotion", "conflict", "state", "plot_function"]:
        assert analysis.get(key)
    analyzed_context = build_analyzed_generation_context(
        "기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
        "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
        "서윤은 폐쇄 연구동에서 손상된 로그와 동생의 이름을 발견한다.",
        analysis,
    )
    for label in ["요약", "감정", "갈등", "상태", "장면 기능"]:
        assert label in analyzed_context

    response = (
        "### 닫힌 문\n\n서윤은 붉은 열쇠를 주머니에 감추었다.\n"
        f"{MEMORY_START}\n"
        '{"section_index":1,"title":"닫힌 문","summary":"서윤이 붉은 열쇠를 숨겼다.",'
        '"characters":["서윤"],"facts":["붉은 열쇠는 서윤이 갖고 있다."],'
        '"open_clues":["열쇠가 여는 문은 아직 모른다."],"resolved_clues":[],"locations":[],'
        '"state_changes":["열쇠 소유자가 서윤으로 바뀌었다."],'
        '"state_updates":[{"entity":"붉은 열쇠","attribute":"owner","value":"서윤"}],'
        '"relations":[{"source":"서윤","relation":"possesses","target":"붉은 열쇠"}],'
        '"keywords":["서윤","붉은 열쇠"]}'
        f"\n{MEMORY_END}"
    )
    prose, memory = split_story_memory(response, 1, ["서윤"])
    assert MEMORY_START not in prose
    assert memory.open_clues
    retrieved_memories = retrieve_story_memories(
        [
            memory,
            StoryMemory(
                section_index=2,
                title="비 내리는 역",
                summary="민재가 열차를 놓쳤다.",
                keywords=["민재", "열차"],
            ),
        ],
        "서윤이 붉은 열쇠로 잠긴 문을 연다.",
        top_k=1,
        current_section=3,
    )
    assert retrieved_memories[0][0].section_index == 1
    ledger = build_story_ledger([memory], group_size=2)
    assert ledger["current_states"][0]["value"] == "서윤"
    assert ledger["relations"][0]["target"] == "붉은 열쇠"
    hierarchical_context, _ = format_hierarchical_story_context(
        [memory],
        retrieved_memories,
        "서윤이 붉은 열쇠를 사용한다.",
        max_chars=2000,
        group_size=2,
    )
    assert "Current state ledger" in hierarchical_context
    assert "Relevant knowledge graph" in hierarchical_context
    visible_chunks: list[str] = []
    stream_filter = StoryMemoryStreamFilter(visible_chunks.append)
    for chunk in [response[:17], response[17:53], response[53:]]:
        stream_filter.feed(chunk)
    stream_filter.finish()
    assert MEMORY_START not in "".join(visible_chunks)
    assert "서윤은 붉은 열쇠" in "".join(visible_chunks)


def test_dry_run_pipeline() -> None:
    with tempfile.TemporaryDirectory(prefix="novel_jepa_smoke_") as tmp:
        config = AppConfig(output_root=tmp)
        config.data.reuse_existing = False
        config.training.epochs = 2
        config.training.batch_size = 2
        config.training.hidden_dim = 128
        config.training.num_layers = 3
        config.training.early_stopping_patience = 0
        ensure_project_dirs(config)
        client = OllamaClient(
            config.ollama.base_url,
            config.ollama.chat_model,
            config.ollama.embed_model,
            dry_run=True,
        )

        generated = generate_synthetic_dataset(config, client, "한국형 SF 미스터리", 4)
        assert generated["written"] == 4
        assert generated["candidate_limit"] >= 4
        assert generated["diversity"]["unique_signatures"] >= 1
        filtered = filter_jsonl(config)
        assert filtered["kept"] >= 2
        samples = [
            json.loads(line)
            for line in resolve_path(config, config.data.filtered_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = diversity_report_from_samples(samples)
        assert report["sample_count"] == filtered["kept"]
        assert report["axes"]["transition_shape"]["unique"] >= 1
        embedded = embed_dataset(config, client)
        assert embedded["count"] >= 2
        with np.load(resolve_path(config, config.data.embeddings_path)) as embedding_payload:
            assert str(embedding_payload["embedding_backend"]) == "dry-run"
        current_index = build_current_context_index(config)
        next_index = build_next_scene_index(config)
        assert current_index.exists()
        assert next_index.exists()
        history = train_predictor(config)
        assert history["best_val_cosine"] > -1
        payload = torch.load(resolve_path(config, config.training.checkpoint_path), map_location="cpu")
        assert payload.get("train_idx")
        assert payload.get("val_idx")
        diagnostics = evaluate_planner_diagnostics(config, top_k=2)
        assert diagnostics["available"]
        assert "validation_pred_target_cosine" in diagnostics
        assert diagnostics.get("validation", {}).get("available")

        predicted = predict_next_embedding(
            config,
            client,
            "서윤은 폐쇄 연구동에서 손상된 로그를 발견한다.",
            world="기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
            characters="서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
        )
        retrieved = retrieve_by_vector(config, predicted, 2)
        assert retrieved
        output = generate_with_jepa(
            config,
            client,
            "기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
            "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
            "서윤은 폐쇄 연구동에서 손상된 로그를 발견한다.",
        )
        assert isinstance(output, str)
        assert output.strip()
        creative_result = generate_with_controlled_hallucination(
            config,
            client,
            "湲곗뼲 ?뷀뼢??臾쇰━???붿쟻?쇰줈 ?⑤뒗 洹쇰????쒖슱.",
            "?쒖쑄: ?숈깮??李얜뒗 湲곕줉 蹂듭썝媛. 誘쇱옱: 吏꾩떎???④릿 ?곌뎄??",
            "?쒖쑄? ?먯뇙 ?곌뎄?숈뿉???먯긽??濡쒓렇瑜?諛쒓껄?쒕떎.",
            return_details=True,
        )
        creative_output = str(creative_result["text"])
        assert isinstance(creative_output, str)
        assert creative_output.strip()
        assert MEMORY_START not in creative_output
        assert creative_output.count("### ") >= 2
        assert resolve_path(config, config.generation.longform_checkpoint_path).exists()
        creative_details = creative_result["planner"]
        assert creative_details["story_memory_count"] >= 2
        assert creative_details["story_memory_retrievals"] >= 1
        assert resolve_path(config, config.generation.story_memory_path).exists()
        assert resolve_path(config, config.generation.story_ledger_path).exists()
        continued_result = generate_with_controlled_hallucination(
            config,
            client,
            "기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
            "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
            "서윤은 폐쇄 연구동에서 손상된 로그와 동생의 이름을 발견한다.",
            continue_existing=True,
            turn_target_chars=5000,
            continuation_instruction="붉은 열쇠의 단서를 따라가되 배신자의 정체는 숨긴다.",
            return_details=True,
        )
        assert len(str(continued_result["text"])) > len(creative_output)
        assert continued_result["planner"]["turn_number"] == 2
        assert continued_result["planner"]["continued_existing"]
        bundle = export_longform_bundle(
            config,
            world="기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
            characters="서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
            previous_scene="서윤은 폐쇄 연구동에서 손상된 로그를 발견한다.",
        )
        imported_config = AppConfig(output_root=str(Path(tmp) / "imported_bundle"))
        ensure_project_dirs(imported_config)
        imported = import_longform_file(
            imported_config,
            "saved_story.zip",
            bundle,
        )
        imported_status = longform_artifact_status(imported_config)
        imported_text = resolve_path(
            imported_config,
            imported_config.generation.longform_checkpoint_path,
        ).read_text(encoding="utf-8")
        original_body = re.sub(r"(?m)^###.*$", "", str(continued_result["text"]))
        imported_body = re.sub(r"(?m)^###.*$", "", imported_text)
        assert re.sub(r"\s+", "", imported_body) == re.sub(r"\s+", "", original_body)
        assert imported_status["section_count"] == continued_result["planner"]["completed_sections"]
        assert imported_status["memory_count"] == continued_result["planner"]["story_memory_count"]
        assert imported["context"]["characters"].startswith("서윤:")
        assert not imported["memory_rebuilt"]

        raw_config = AppConfig(output_root=str(Path(tmp) / "imported_markdown"))
        ensure_project_dirs(raw_config)
        raw_import = import_longform_file(
            raw_config,
            "draft.md",
            str(continued_result["text"]).encode("utf-8"),
            characters="서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
        )
        assert raw_import["memory_rebuilt"]
        assert raw_import["section_count"] == imported_status["section_count"]
        rag_output = generate_with_rag(
            config,
            client,
            "기억 잔향이 물리적 흔적으로 남는 근미래 서울.",
            "서윤: 동생을 찾는 기록 복원가. 민재: 진실을 숨긴 연구원.",
            "서윤은 폐쇄 연구동에서 손상된 로그를 발견한다.",
        )
        assert isinstance(rag_output, str)
        assert rag_output.strip()


if __name__ == "__main__":
    test_builders_and_loss()
    test_dry_run_pipeline()
    print("JEPA-inspired smoke checks passed.")
