from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.share_artifact import export_version, import_bundle  # noqa: E402
from src.service.artifacts import ActiveModelUnavailable, load_active_manifest, sha256_file  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402


VERSION = "20260806T111529Z-7262bfe2"
ARTIFACT_FILES = {
    "dataset": "dataset.jsonl",
    "embeddings": "scenes.npz",
    "current_index": "current_context.faiss",
    "next_index": "next_scene.faiss",
    "checkpoint": "predictor.pt",
    "model_card": "model_card.json",
    "history": "train_history.json",
    "diagnostics": "diagnostics.json",
}


def build_version(config: AppConfig, *, foreign_root: str) -> Path:
    """Write a version whose manifest records another machine's absolute paths."""
    root = resolve_path(config, config.consumer.versions_dir) / VERSION
    root.mkdir(parents=True, exist_ok=True)
    for key, name in ARTIFACT_FILES.items():
        (root / name).write_text(f"content for {key}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "status": "active",
        "models": {"chat": config.consumer.chat_model, "embedding": config.consumer.embed_model},
        "paths": {key: f"{foreign_root}\\{name}" for key, name in ARTIFACT_FILES.items()},
        "file_hashes": {key: sha256_file(root / name) for key, name in ARTIFACT_FILES.items()},
        # Without this the loader refuses at the gate check and never reaches the
        # path handling these tests exist to cover.
        "quality_gate": {"passed": True, "checks": [], "failures": []},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class ManifestPortabilityTests(unittest.TestCase):
    """A manifest built elsewhere must still load here."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.root = build_version(
            self.config, foreign_root=r"D:\someone-elses-checkout\artifacts\versions\%s" % VERSION
        )
        active = resolve_path(self.config, self.config.consumer.active_manifest_path)
        active.parent.mkdir(parents=True, exist_ok=True)
        active.write_text((self.root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_foreign_absolute_paths_still_resolve(self) -> None:
        manifest = load_active_manifest(self.config, verify_files=True)

        self.assertEqual(manifest["version"], VERSION)

    def test_a_missing_file_is_still_caught(self) -> None:
        (self.root / "predictor.pt").unlink()

        with self.assertRaises(ActiveModelUnavailable):
            load_active_manifest(self.config)

    def test_a_modified_file_is_still_caught(self) -> None:
        (self.root / "predictor.pt").write_text("tampered", encoding="utf-8")

        with self.assertRaises(ActiveModelUnavailable):
            load_active_manifest(self.config, verify_files=True)

    def test_a_path_that_tries_to_escape_is_rejected(self) -> None:
        active = resolve_path(self.config, self.config.consumer.active_manifest_path)
        manifest = json.loads(active.read_text(encoding="utf-8"))
        manifest["paths"]["checkpoint"] = ".."
        active.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(ActiveModelUnavailable):
            load_active_manifest(self.config)


class ShareRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sender = tempfile.TemporaryDirectory()
        self.receiver = tempfile.TemporaryDirectory()
        self.bundles = tempfile.TemporaryDirectory()
        self.sender_config = AppConfig(output_root=self.sender.name)
        self.receiver_config = AppConfig(output_root=self.receiver.name)
        build_version(self.sender_config, foreign_root=r"C:\sender\artifacts\versions\%s" % VERSION)
        self.archive = Path(self.bundles.name) / "bundle.zip"

    def tearDown(self) -> None:
        for temporary in (self.sender, self.receiver, self.bundles):
            temporary.cleanup()

    def test_a_version_survives_export_and_import(self) -> None:
        export_version(self.sender_config, VERSION, self.archive)
        import_bundle(self.receiver_config, self.archive, activate=True)

        installed = resolve_path(self.receiver_config, self.receiver_config.consumer.versions_dir) / VERSION
        for name in ARTIFACT_FILES.values():
            self.assertTrue((installed / name).exists(), f"{name} was not installed")
        manifest = load_active_manifest(self.receiver_config, verify_files=True)
        self.assertEqual(manifest["version"], VERSION)

    def test_import_without_activate_leaves_no_active_model(self) -> None:
        export_version(self.sender_config, VERSION, self.archive)
        import_bundle(self.receiver_config, self.archive, activate=False)

        active = resolve_path(self.receiver_config, self.receiver_config.consumer.active_manifest_path)
        self.assertFalse(active.exists())

    def test_a_corrupted_bundle_is_refused_and_leaves_nothing_behind(self) -> None:
        import zipfile

        export_version(self.sender_config, VERSION, self.archive)
        tampered = Path(self.bundles.name) / "tampered.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(tampered, "w") as target:
            for name in source.namelist():
                data = source.read(name)
                if name.endswith("predictor.pt"):
                    data = b"corrupted in transit"
                target.writestr(name, data)

        with self.assertRaises(SystemExit):
            import_bundle(self.receiver_config, tampered, activate=True)

        versions = resolve_path(self.receiver_config, self.receiver_config.consumer.versions_dir)
        self.assertFalse((versions / VERSION).exists())
        self.assertEqual([p for p in versions.glob(".incoming-*")], [])

    def test_reinstalling_over_an_existing_version_is_refused(self) -> None:
        export_version(self.sender_config, VERSION, self.archive)
        import_bundle(self.receiver_config, self.archive, activate=False)

        with self.assertRaises(SystemExit):
            import_bundle(self.receiver_config, self.archive, activate=False)


if __name__ == "__main__":
    unittest.main()
