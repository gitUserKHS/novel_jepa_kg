from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.planner.model import MLPPredictor, count_parameters
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def _training_config_dict(config: AppConfig) -> dict[str, Any]:
    if hasattr(config.training, "model_dump"):
        return config.training.model_dump()
    return config.training.dict()


def predictor_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine_component = 1 - torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
    mse_component = torch.nn.functional.mse_loss(
        torch.nn.functional.normalize(pred, dim=1),
        torch.nn.functional.normalize(target, dim=1),
    )
    return cosine_component + 0.05 * mse_component, cosine_component, mse_component


def _make_model(dim: int, config_dict: dict[str, Any]) -> MLPPredictor:
    model_type = str(config_dict.get("model_type", "mlp"))
    residual = model_type == "residual_mlp" or int(config_dict.get("num_layers", 2)) > 2
    return MLPPredictor(
        dim=dim,
        hidden_dim=config_dict.get("hidden_dim"),
        num_layers=int(config_dict.get("num_layers", 2)),
        dropout=float(config_dict.get("dropout", 0.0)),
        residual=residual,
    )


ProgressCallback = Callable[[dict[str, Any]], None]


def train_predictor(config: AppConfig, progress_callback: ProgressCallback | None = None) -> dict:
    torch.manual_seed(config.project.seed)
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)

    embeddings_path = resolve_path(config, config.data.embeddings_path)
    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    history_path = resolve_path(config, "reports/runs/latest_train_history.json")
    model_card_path = resolve_path(config, "checkpoints/predictor/model_card.json")
    ensure_parent(checkpoint_path)
    ensure_parent(history_path)
    ensure_parent(model_card_path)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]),
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_x = x[val_idx].to(device)
    val_y = y[val_idx].to(device)

    train_config = _training_config_dict(config)
    model = _make_model(dim=x.shape[1], config_dict=train_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = device.type == "cuda" and config.training.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = -1.0
    best_epoch = 0
    stale_epochs = 0
    epochs = []
    param_count = count_parameters(model)
    logger.info("Training predictor for %s epochs on %s with %s params", config.training.epochs, device, param_count)

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        losses = []
        cosine_losses = []
        mse_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(batch_x)
                loss, cosine_component, mse_component = predictor_loss(pred, batch_y)
            scaler.scale(loss).backward()
            if config.training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            cosine_losses.append(float(cosine_component.detach().cpu()))
            mse_losses.append(float(mse_component.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss, val_cosine_loss, val_mse = predictor_loss(val_pred, val_y)
            val_cosine = torch.nn.functional.cosine_similarity(val_pred, val_y, dim=1).mean().item()
        train_loss = float(np.mean(losses)) if losses else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cosine_loss": float(np.mean(cosine_losses)) if cosine_losses else 0.0,
            "train_norm_mse": float(np.mean(mse_losses)) if mse_losses else 0.0,
            "val_loss": float(val_loss.detach().cpu()),
            "val_cosine_loss": float(val_cosine_loss.detach().cpu()),
            "val_norm_mse": float(val_mse.detach().cpu()),
            "val_cosine": float(val_cosine),
        }
        epochs.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    **row,
                    "total_epochs": config.training.epochs,
                    "best_val_cosine": max(best_val, float(val_cosine)),
                    "parameter_count": param_count,
                    "device": str(device),
                }
            )
        if val_cosine > best_val:
            best_val = float(val_cosine)
            best_epoch = epoch
            stale_epochs = 0
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "checkpoint_version": 2,
                    "model_state": cpu_state,
                    "dim": int(x.shape[1]),
                    "config": train_config,
                    "parameter_count": param_count,
                    "best_val_cosine": best_val,
                    "best_epoch": best_epoch,
                    "embedding_path": str(embeddings_path),
                    "dataset_size": int(len(x)),
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if config.training.early_stopping_patience > 0 and stale_epochs >= config.training.early_stopping_patience:
                break

    trained_at = datetime.now().isoformat(timespec="seconds")
    history = {
        "trained_at": trained_at,
        "best_val_cosine": best_val,
        "best_epoch": best_epoch,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "parameter_count": param_count,
        "checkpoint_path": str(checkpoint_path),
        "embedding_path": str(embeddings_path),
        "dataset_size": int(len(x)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "training_config": train_config,
        "epochs": epochs,
    }
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["training_history"] = history
        payload["trained_at"] = trained_at
        torch.save(payload, checkpoint_path)
    model_card = {
        key: value
        for key, value in history.items()
        if key not in {"epochs"}
    }
    model_card["epoch_count"] = len(epochs)
    model_card_path.write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def load_predictor(checkpoint_path: Path) -> MLPPredictor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    config_dict = payload.get("config", {})
    model = _make_model(dim=int(payload["dim"]), config_dict=config_dict)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
