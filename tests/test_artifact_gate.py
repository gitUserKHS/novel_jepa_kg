from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.service.artifacts import promote_candidate, quality_gate
from src.service.consumer_store import ConsumerStore
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


def passing_facts() -> dict[str, object]:
    return {
        "sample_count": 40,
        "validation_count": 6,
        "validation_cosine": 0.70,
        "hit_at_5": 0.60,
        "normalized_top1_diversity": 0.50,
        "effective_rank_ratio": 0.70,
        "jepa_score": 0.83,
        "rag_next_score": 0.75,
        "file_fingerprint_match": True,
        "vector_dimensions_match": True,
        "model_names_match": True,
    }


class ArtifactGateTests(unittest.TestCase):
    def test_gate_rejects_current_ten_sample_scale(self) -> None:
        config = AppConfig()
        facts = passing_facts()
        facts.update({"sample_count": 10, "validation_count": 1})
        report = quality_gate(facts, config.consumer)
        self.assertFalse(report["passed"])
        self.assertEqual(
            {row["name"] for row in report["failures"]},
            {"sample_count", "validation_count"},
        )

    def test_gate_accepts_all_required_thresholds(self) -> None:
        report = quality_gate(passing_facts(), AppConfig().consumer)
        self.assertTrue(report["passed"])

    def test_failed_promotion_preserves_existing_active_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            store = ConsumerStore(config)
            store.set_maintenance(True)
            active = resolve_path(config, config.consumer.active_manifest_path)
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text('{"version":"old"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                promote_candidate(config, {"version": "bad", "quality_gate": {"passed": False}})
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["version"], "old")

    def test_passing_promotion_atomically_replaces_active_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            store = ConsumerStore(config)
            store.set_maintenance(True)
            version = "candidate-v1"
            candidate_root = resolve_path(config, config.consumer.candidates_dir) / version
            candidate_root.mkdir(parents=True)
            paths = {}
            for key, name in {
                "dataset": "dataset.jsonl",
                "embeddings": "scenes.npz",
                "current_index": "current.faiss",
                "next_index": "next.faiss",
                "checkpoint": "predictor.pt",
                "model_card": "model_card.json",
                "history": "history.json",
                "diagnostics": "diagnostics.json",
            }.items():
                path = candidate_root / name
                path.write_bytes(b"test")
                paths[key] = str(path)
            gate = quality_gate(passing_facts(), config.consumer)
            manifest = {
                "version": version,
                "status": "candidate",
                "paths": paths,
                "models": {"chat": config.consumer.chat_model, "embedding": config.consumer.embed_model},
                "dataset_sha256": "dataset",
                "file_hashes": {},
                "facts": passing_facts(),
                "quality_gate": gate,
            }
            inspected = {
                "available": True,
                "facts": passing_facts(),
            }
            with patch("src.service.artifacts.inspect_artifact_set", return_value=inspected):
                promoted = promote_candidate(config, manifest)
            active = resolve_path(config, config.consumer.active_manifest_path)
            self.assertEqual(promoted["status"], "active")
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["version"], version)
            self.assertFalse(candidate_root.exists())
            self.assertTrue((resolve_path(config, config.consumer.versions_dir) / version).exists())


if __name__ == "__main__":
    unittest.main()
