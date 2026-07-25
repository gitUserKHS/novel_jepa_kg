from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.service.health import model_is_available, normalize_model_name
from src.service.job_lock import ServiceBusyError, acquire_project_job
from src.service.security import (
    access_policy,
    generate_secret,
    hash_story_secret,
    token_fingerprint,
    verify_access_token,
    verify_story_secret,
)
from src.utils.config import AppConfig, apply_environment_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_checked_in_default_uses_gemma4_e4b(self) -> None:
        raw = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
        config = AppConfig(**raw)
        self.assertEqual(config.ollama.chat_model, "gemma4:e4b")
        self.assertEqual(config.service.bind_host, "127.0.0.1")
        self.assertEqual(config.service.port, 8502)
        self.assertEqual(config.consumer.bind_host, "0.0.0.0")
        self.assertEqual(config.consumer.port, 8501)
        self.assertEqual(config.consumer.chat_model, "gemma4:e4b")
        self.assertEqual(config.consumer.embed_model, "embeddinggemma:latest")
        self.assertEqual(config.consumer.target_char_options()[0], 10000)
        self.assertIn(42000, config.consumer.target_char_options())
        self.assertEqual(config.consumer.target_char_options()[-1], 50000)
        self.assertFalse(config.service.require_access_token)

    def test_environment_overrides_runtime_settings(self) -> None:
        values = {
            "NOVEL_JEPA_CHAT_MODEL": "gemma4:e4b-test",
            "NOVEL_JEPA_BIND_HOST": "0.0.0.0",
            "NOVEL_JEPA_PORT": "9510",
            "NOVEL_JEPA_REQUIRE_AUTH": "true",
        }
        with patch.dict(os.environ, values, clear=True):
            config = apply_environment_overrides(AppConfig())
        self.assertEqual(config.ollama.chat_model, "gemma4:e4b-test")
        self.assertEqual(config.service.bind_host, "0.0.0.0")
        self.assertEqual(config.service.port, 9510)
        self.assertTrue(config.service.require_access_token)

    def test_invalid_boolean_override_is_rejected(self) -> None:
        with patch.dict(os.environ, {"NOVEL_JEPA_REQUIRE_AUTH": "sometimes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Invalid boolean"):
                apply_environment_overrides(AppConfig())


class ServiceTests(unittest.TestCase):
    def test_access_policy_requires_explicit_enablement(self) -> None:
        policy = access_policy(False, "TOKEN", {"TOKEN": "secret"})
        self.assertFalse(policy.required)
        self.assertTrue(policy.configured)

        required = access_policy(True, "TOKEN", {})
        self.assertTrue(required.required)
        self.assertFalse(required.configured)

    def test_token_comparison_and_fingerprint(self) -> None:
        self.assertTrue(verify_access_token("secret", "secret"))
        self.assertFalse(verify_access_token("secret", "other"))
        self.assertFalse(verify_access_token("", ""))
        self.assertEqual(token_fingerprint("secret"), token_fingerprint("secret"))
        self.assertNotEqual(token_fingerprint("secret"), token_fingerprint("other"))

    def test_story_secret_uses_random_scrypt_hash(self) -> None:
        secret = generate_secret()
        first_salt, first_hash = hash_story_secret(secret)
        second_salt, second_hash = hash_story_secret(secret)
        self.assertNotEqual(first_salt, second_salt)
        self.assertNotEqual(first_hash, second_hash)
        self.assertTrue(verify_story_secret(secret, first_salt, first_hash))
        self.assertFalse(verify_story_secret("wrong", first_salt, first_hash))

    def test_ollama_model_matching_accepts_latest_alias(self) -> None:
        self.assertEqual(normalize_model_name("EmbeddingGemma:latest"), "embeddinggemma")
        self.assertTrue(model_is_available("gemma4:e4b", ["gemma4:e4b", "other:latest"]))
        self.assertTrue(model_is_available("embeddinggemma", ["embeddinggemma:latest"]))
        self.assertFalse(model_is_available("gemma4:e4b", ["gemma4:12b"]))

    def test_shared_artifact_job_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            first = acquire_project_job(config, "training")
            try:
                with self.assertRaisesRegex(ServiceBusyError, "training"):
                    acquire_project_job(config, "generation")
            finally:
                first.release()

            with acquire_project_job(config, "generation") as second:
                self.assertEqual(second.operation, "generation")


if __name__ == "__main__":
    unittest.main()
