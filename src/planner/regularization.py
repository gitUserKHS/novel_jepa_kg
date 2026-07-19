from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DistributionRegularization:
    loss: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor
    active: bool
    reason: str = ""


def target_visreg_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_slices: int = 64,
    variance_floor_ratio: float = 0.75,
    center_weight: float = 0.1,
    eps: float = 1e-6,
) -> DistributionRegularization:
    """Match a predictor batch to the fixed target-embedding distribution.

    This is VISReg-inspired rather than the paper's exact isotropic-Gaussian
    objective. The project predicts frozen, L2-normalized text embeddings, so
    forcing unit variance in every output dimension would conflict with the
    target manifold. Scale and sliced-Wasserstein terms are therefore measured
    relative to the target batch.
    """
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same [batch, dimension] shape")
    if prediction.shape[0] < 2:
        zero = prediction.sum() * 0.0
        return DistributionRegularization(zero, zero, zero, zero, False, "batch_too_small")

    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    target_mean = target.mean(dim=0)
    prediction_centered = prediction - prediction.mean(dim=0)
    target_centered = target - target_mean

    target_std = target_centered.pow(2).mean(dim=0).add(eps).sqrt().detach()
    prediction_std = prediction_centered.pow(2).mean(dim=0).add(eps).sqrt()
    scale_ratio = prediction_std / target_std
    scale_loss = F.relu(float(variance_floor_ratio) - scale_ratio).pow(2).mean()

    center_loss = ((prediction.mean(dim=0) - target_mean) / target_std).pow(2).mean()

    slice_count = max(1, int(num_slices))
    directions = torch.randn(
        prediction.shape[1],
        slice_count,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    directions = F.normalize(directions, dim=0)
    prediction_projection = prediction_centered @ directions
    target_projection = target_centered @ directions
    projection_scale = target_projection.pow(2).mean(dim=0).add(eps).sqrt().detach()
    prediction_sorted = torch.sort(prediction_projection / projection_scale, dim=0).values
    target_sorted = torch.sort(target_projection / projection_scale, dim=0).values
    shape_loss = F.mse_loss(prediction_sorted, target_sorted)

    total = scale_loss + shape_loss + float(center_weight) * center_loss
    return DistributionRegularization(total, center_loss, scale_loss, shape_loss, True)


def normalized_effective_rank(vectors: torch.Tensor, eps: float = 1e-8) -> float:
    """Return entropy-based effective rank divided by the maximum sample rank."""
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 1:
        return 0.0
    centered = vectors.detach().float() - vectors.detach().float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total) <= eps:
        return 0.0
    probabilities = energy / total
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    effective = entropy.exp()
    maximum = max(1, min(vectors.shape[0] - 1, vectors.shape[1]))
    return float((effective / maximum).clamp(0.0, 1.0).cpu())
