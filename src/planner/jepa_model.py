from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 4,
        dropout: float = 0.1,
        predict_delta: bool = True,
        normalize_prediction: bool = True,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(256, dim * 2)
        block_count = max(1, num_layers - 2)
        self.predict_delta = predict_delta
        self.normalize_prediction = normalize_prediction
        self.input_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden, dropout) for _ in range(block_count)])
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(context_embedding)
        delta_or_target = self.output_projection(self.blocks(hidden))
        predicted = context_embedding + delta_or_target if self.predict_delta else delta_or_target
        if self.normalize_prediction:
            predicted = F.normalize(predicted, dim=1)
        return predicted


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
