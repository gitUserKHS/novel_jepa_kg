from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
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


class MLPPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(128, dim * 2)
        if not residual:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            return

        block_count = max(1, num_layers - 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            *[ResidualBlock(hidden, dropout) for _ in range(block_count)],
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
