from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from src.planner.jepa_train import train_predictor
from src.planner.regularization import normalized_effective_rank, target_visreg_loss
from src.utils.config import AppConfig


class TargetVisRegTests(unittest.TestCase):
    def test_matching_target_distribution_has_lower_loss_than_collapse(self) -> None:
        torch.manual_seed(7)
        target = F.normalize(torch.randn(16, 32), dim=1)
        matching = target.clone()
        collapsed = F.normalize(torch.ones_like(target), dim=1)

        matching_loss = target_visreg_loss(matching, target, num_slices=32)
        collapsed_loss = target_visreg_loss(collapsed, target, num_slices=32)

        self.assertTrue(matching_loss.active)
        self.assertLess(float(matching_loss.loss), 1e-6)
        self.assertGreater(float(collapsed_loss.loss), float(matching_loss.loss) + 0.1)

    def test_effective_rank_exposes_collapsed_predictions(self) -> None:
        diverse = torch.eye(8, 16)
        collapsed = torch.ones(8, 16)

        self.assertGreater(normalized_effective_rank(diverse), 0.8)
        self.assertEqual(normalized_effective_rank(collapsed), 0.0)

    def test_training_records_regularization_and_rank_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding_path = root / "embeddings.npz"
            rng = np.random.default_rng(4)
            current = rng.normal(size=(40, 16)).astype("float32")
            current /= np.linalg.norm(current, axis=1, keepdims=True)
            target = current + rng.normal(scale=0.08, size=current.shape).astype("float32")
            target /= np.linalg.norm(target, axis=1, keepdims=True)
            np.savez(
                embedding_path,
                current_embeddings=current,
                next_embeddings=target,
            )

            config = AppConfig(output_root=str(root))
            config.data.embeddings_path = "embeddings.npz"
            config.training.checkpoint_path = "checkpoint.pt"
            config.training.history_path = "history.json"
            config.training.model_card_path = "model_card.json"
            config.training.epochs = 2
            config.training.batch_size = 16
            config.training.hidden_dim = 32
            config.training.num_layers = 2
            config.training.dropout = 0.0
            config.training.use_context_dropout = False
            config.training.regularization_num_slices = 8
            config.training.early_stopping_patience = 0

            with patch("torch.cuda.is_available", return_value=False):
                history = train_predictor(config)
            diagnostics = history["representation_regularization"]

            self.assertTrue(diagnostics["regularization_enabled"])
            self.assertIn("effective_rank_ratio", diagnostics)
            self.assertTrue((root / "checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
