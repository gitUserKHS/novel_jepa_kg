from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.planner.jepa_model import JEPAPredictor, count_parameters
from src.planner.model import MLPPredictor
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)
ProgressCallback = Callable[[dict[str, Any]], None]


def _training_config_dict(config: AppConfig) -> dict[str, Any]:
    if hasattr(config.training, "model_dump"):
        return config.training.model_dump()
    return config.training.dict()


def representation_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.05,
    norm_weight: float = 0.001,
    normalize_prediction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_norm_weight = 0.0 if normalize_prediction else norm_weight
    cosine_component = 1 - F.cosine_similarity(pred, target, dim=1).mean()
    mse_component = F.mse_loss(F.normalize(pred, dim=1), F.normalize(target, dim=1))
    norm_component = (pred.norm(dim=1) - target.norm(dim=1).detach()).pow(2).mean()
    total = cosine_component + mse_weight * mse_component + effective_norm_weight * norm_component
    return total, cosine_component, mse_component, norm_component


def predictor_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total, cosine_component, mse_component, _norm_component = representation_prediction_loss(pred, target)
    return total, cosine_component, mse_component


def _make_jepa_model(dim: int, config_dict: dict[str, Any]) -> JEPAPredictor:
    return JEPAPredictor(
        dim=dim,
        hidden_dim=config_dict.get("hidden_dim"),
        num_layers=int(config_dict.get("num_layers", 4)),
        dropout=float(config_dict.get("dropout", 0.1)),
        predict_delta=bool(config_dict.get("predict_delta", True)),
        normalize_prediction=bool(config_dict.get("normalize_prediction", True)),
    )


def _make_legacy_model(dim: int, config_dict: dict[str, Any]) -> MLPPredictor:
    model_type = str(config_dict.get("model_type", "mlp"))
    residual = model_type == "residual_mlp" or int(config_dict.get("num_layers", 2)) > 2
    return MLPPredictor(
        dim=dim,
        hidden_dim=config_dict.get("hidden_dim"),
        num_layers=int(config_dict.get("num_layers", 2)),
        dropout=float(config_dict.get("dropout", 0.0)),
        residual=residual,
    )


def _load_embedding_arrays(config: AppConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    embeddings_path = resolve_path(config, config.data.embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    data = np.load(embeddings_path)
    x = np.asarray(data["current_embeddings"], dtype="float32")
    y = np.asarray(data["next_embeddings"], dtype="float32")
    dropout_x = None
    augmented_count = 0
    if config.training.use_context_dropout and "dropout_context_embeddings" in data:
        candidate = np.asarray(data["dropout_context_embeddings"], dtype="float32")
        if candidate.shape == x.shape:
            dropout_x = candidate
            augmented_count = int(len(dropout_x))
    metadata = {
        "embedding_path": str(embeddings_path),
        "base_pair_count": int(len(data["current_embeddings"])),
        "augmented_context_count": augmented_count,
        "target_count": int(len(data["next_embeddings"])),
    }
    return x, y, dropout_x, metadata


def train_predictor(config: AppConfig, progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    torch.manual_seed(config.project.seed)
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)

    checkpoint_path = resolve_path(config, config.training.checkpoint_path)
    history_path = resolve_path(config, "reports/runs/latest_train_history.json")
    model_card_path = resolve_path(config, "checkpoints/predictor/model_card.json")
    ensure_parent(checkpoint_path)
    ensure_parent(history_path)
    ensure_parent(model_card_path)

    base_x, base_y, dropout_x, data_metadata = _load_embedding_arrays(config)
    if len(base_x) < 2:
        raise ValueError("At least two embedding pairs are required for train/validation split.")

    indices = torch.randperm(len(base_x))
    val_size = max(1, int(len(base_x) * config.training.val_ratio))
    if val_size >= len(base_x):
        val_size = 1
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = indices[:1]
        val_idx = indices[1:] if len(indices) > 1 else indices[:1]

    train_x_np = base_x[train_idx.numpy()]
    train_y_np = base_y[train_idx.numpy()]
    if dropout_x is not None:
        train_x_np = np.concatenate([train_x_np, dropout_x[train_idx.numpy()]], axis=0)
        train_y_np = np.concatenate([train_y_np, base_y[train_idx.numpy()]], axis=0)
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    val_x_cpu = torch.tensor(base_x[val_idx.numpy()], dtype=torch.float32)
    val_y_cpu = torch.tensor(base_y[val_idx.numpy()], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_x = val_x_cpu.to(device)
    val_y = val_y_cpu.to(device)

    train_config = _training_config_dict(config)
    model = _make_jepa_model(dim=base_x.shape[1], config_dict=train_config).to(device)
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
    early_stopped = False
    stopped_reason = "max_epochs"
    epochs: list[dict[str, Any]] = []
    param_count = count_parameters(model)
    mse_weight = float(config.training.loss_mse_weight)
    norm_weight = float(config.training.loss_norm_weight)
    normalize_prediction = bool(config.training.normalize_prediction)
    effective_norm_weight = 0.0 if normalize_prediction else norm_weight
    logger.info("Training JEPA-inspired predictor for %s epochs on %s with %s params", config.training.epochs, device, param_count)

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        losses: list[float] = []
        cosine_losses: list[float] = []
        mse_losses: list[float] = []
        norm_losses: list[float] = []
        pred_norms: list[float] = []
        pred_cosines: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(batch_x)
                loss, cosine_component, mse_component, norm_component = representation_prediction_loss(
                    pred,
                    batch_y,
                    mse_weight=mse_weight,
                    norm_weight=norm_weight,
                    normalize_prediction=normalize_prediction,
                )
            scaler.scale(loss).backward()
            if config.training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            cosine_losses.append(float(cosine_component.detach().cpu()))
            mse_losses.append(float(mse_component.detach().cpu()))
            norm_losses.append(float(norm_component.detach().cpu()))
            pred_norms.append(float(pred.norm(dim=1).mean().detach().cpu()))
            pred_cosines.append(float(F.cosine_similarity(pred.detach(), batch_y, dim=1).mean().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss, val_cosine_loss, val_mse, val_norm = representation_prediction_loss(
                val_pred,
                val_y,
                mse_weight=mse_weight,
                norm_weight=norm_weight,
                normalize_prediction=normalize_prediction,
            )
            val_cosine = F.cosine_similarity(val_pred, val_y, dim=1).mean().item()
            val_pred_norm = val_pred.norm(dim=1).mean().item()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_cosine_loss": float(np.mean(cosine_losses)) if cosine_losses else 0.0,
            "train_norm_mse": float(np.mean(mse_losses)) if mse_losses else 0.0,
            "train_norm_regularization": float(np.mean(norm_losses)) if norm_losses else 0.0,
            "train_pred_target_cosine": float(np.mean(pred_cosines)) if pred_cosines else 0.0,
            "train_predicted_vector_norm": float(np.mean(pred_norms)) if pred_norms else 0.0,
            "effective_loss_norm_weight": effective_norm_weight,
            "val_loss": float(val_loss.detach().cpu()),
            "val_cosine_loss": float(val_cosine_loss.detach().cpu()),
            "val_norm_mse": float(val_mse.detach().cpu()),
            "val_norm_regularization": float(val_norm.detach().cpu()),
            "val_cosine": float(val_cosine),
            "val_pred_target_cosine": float(val_cosine),
            "val_predicted_vector_norm": float(val_pred_norm),
        }
        improved = val_cosine > best_val
        should_stop = False
        if improved:
            best_val = float(val_cosine)
            best_epoch = epoch
            stale_epochs = 0
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "checkpoint_version": 3,
                    "predictor_class": "JEPAPredictor",
                    "model_state": cpu_state,
                    "dim": int(base_x.shape[1]),
                    "config": train_config,
                    "effective_loss_norm_weight": effective_norm_weight,
                    "parameter_count": param_count,
                    "best_val_cosine": best_val,
                    "best_epoch": best_epoch,
                    "data_metadata": data_metadata,
                    "train_idx": train_idx.cpu().numpy().astype("int64").tolist(),
                    "val_idx": val_idx.cpu().numpy().astype("int64").tolist(),
                    "dataset_size": int(len(base_x)),
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "augmented_train_size": int(len(train_x)),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if config.training.early_stopping_patience > 0 and stale_epochs >= config.training.early_stopping_patience:
                early_stopped = True
                stopped_reason = f"early_stopping_patience_{config.training.early_stopping_patience}"
                should_stop = True
        row.update(
            {
                "is_best": improved,
                "best_val_cosine": best_val,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "early_stopping_patience": int(config.training.early_stopping_patience),
                "early_stop_triggered": should_stop,
            }
        )
        epochs.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    **row,
                    "total_epochs": config.training.epochs,
                    "parameter_count": param_count,
                    "device": str(device),
                }
            )
        if should_stop:
            break

    trained_at = datetime.now().isoformat(timespec="seconds")
    history = {
        "trained_at": trained_at,
        "planner_type": "JEPA-inspired latent transition planner",
        "best_val_cosine": best_val,
        "best_pred_target_cosine": best_val,
        "best_epoch": best_epoch,
        "requested_epochs": int(config.training.epochs),
        "completed_epochs": len(epochs),
        "early_stopped": early_stopped,
        "stopped_reason": stopped_reason,
        "early_stopping_patience": int(config.training.early_stopping_patience),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "parameter_count": param_count,
        "checkpoint_path": str(checkpoint_path),
        **data_metadata,
        "effective_loss_norm_weight": effective_norm_weight,
        "dataset_size": int(len(base_x)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "augmented_train_size": int(len(train_x)),
        "training_config": train_config,
        "epochs": epochs,
    }
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["training_history"] = history
        payload["trained_at"] = trained_at
        torch.save(payload, checkpoint_path)
    model_card = {key: value for key, value in history.items() if key not in {"epochs"}}
    model_card["epoch_count"] = len(epochs)
    model_card_path.write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def load_predictor(checkpoint_path: Path) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    config_dict = payload.get("config", {})
    if int(payload.get("checkpoint_version", 1)) >= 3 or payload.get("predictor_class") == "JEPAPredictor":
        model = _make_jepa_model(dim=int(payload["dim"]), config_dict=config_dict)
    else:
        model = _make_legacy_model(dim=int(payload["dim"]), config_dict=config_dict)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
