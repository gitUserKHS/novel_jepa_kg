from __future__ import annotations

import numpy as np
import torch

from src.llm.ollama_client import OllamaClient
from src.planner.train import load_predictor
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def predict_next_embedding(config: AppConfig, client: OllamaClient, previous_scene: str) -> np.ndarray:
    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    model = load_predictor(checkpoint_path)
    current = client.embed([previous_scene])
    with torch.no_grad():
        pred = model(torch.tensor(current, dtype=torch.float32)).detach().cpu().numpy()[0]
    return pred.astype("float32")
