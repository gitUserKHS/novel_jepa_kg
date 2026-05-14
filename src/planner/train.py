from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.planner.model import MLPPredictor
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()


def train_predictor(config: AppConfig) -> dict:
    torch.manual_seed(config.project.seed)
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)

    embeddings_path = resolve_path(config, config.data.embeddings_path)
    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    history_path = resolve_path(config, "reports/runs/latest_train_history.json")
    ensure_parent(checkpoint_path)
    ensure_parent(history_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path)
    x = torch.tensor(data["current_embeddings"], dtype=torch.float32)
    y = torch.tensor(data["next_embeddings"], dtype=torch.float32)
    if len(x) < 2:
        raise ValueError("At least two embedding pairs are required for train/validation split.")

    indices = torch.randperm(len(x))
    val_size = max(1, int(len(x) * config.training.val_ratio))
    if val_size >= len(x):
        val_size = 1
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices[:1]
        val_idx = indices[1:] if len(indices) > 1 else indices[:1]

    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_x = x[val_idx]
    val_y = y[val_idx]

    model = MLPPredictor(dim=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    best_val = -1.0
    epochs = []
    logger.info("Training predictor for %s epochs", config.training.epochs)
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = cosine_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_cosine = torch.nn.functional.cosine_similarity(val_pred, val_y, dim=1).mean().item()
        train_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "train_loss": train_loss, "val_cosine": float(val_cosine)}
        epochs.append(row)
        if val_cosine > best_val:
            best_val = float(val_cosine)
            torch.save(
                {"model_state": model.state_dict(), "dim": int(x.shape[1]), "config": config.training.dict()},
                checkpoint_path,
            )
    history = {"best_val_cosine": best_val, "epochs": epochs}
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def load_predictor(checkpoint_path: Path) -> MLPPredictor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = MLPPredictor(dim=int(payload["dim"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
