from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filter_dataset import filter_jsonl
from src.data.generate_synthetic import generate_synthetic_dataset
from src.embedding.embed_scenes import embed_dataset
from src.embedding.vector_store import build_next_scene_index, retrieve_by_vector
from src.generation.generate_with_jepa import generate_with_jepa
from src.llm.ollama_client import OllamaClient
from src.planner.jepa_dataset import MASK_TOKEN, build_context_text, build_target_text
from src.planner.jepa_model import JEPAPredictor
from src.planner.jepa_train import representation_prediction_loss, train_predictor
from src.planner.predict import evaluate_planner_diagnostics, predict_next_embedding
from src.utils.config import AppConfig
from src.utils.paths import ensure_project_dirs


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
        filtered = filter_jsonl(config)
        assert filtered["kept"] >= 2
        embedded = embed_dataset(config, client)
        assert embedded["count"] >= 2
        build_next_scene_index(config)
        history = train_predictor(config)
        assert history["best_val_cosine"] > -1
        diagnostics = evaluate_planner_diagnostics(config, top_k=2)
        assert diagnostics["available"]
        assert "pred_target_cosine" in diagnostics

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


if __name__ == "__main__":
    test_builders_and_loss()
    test_dry_run_pipeline()
    print("JEPA-inspired smoke checks passed.")
