from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generation.hallucination import generate_with_controlled_hallucination
from src.utils.config import AppConfig


class FakeClient:
    dry_run = False

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        if self.calls == 1:
            return "### 끊긴 초고\n\n문장이 너무 짧게"
        body = (
            "서윤은 흩어진 기록의 시간을 맞춘 뒤 잠긴 문을 열었다. "
            "그 선택으로 복도의 조명이 켜졌고, 다음 방으로 이어지는 길이 드러났다. "
        ) * 18
        return f"### 이어진 선택\n\n{body.strip()}"


class LongformStabilityRetryTests(unittest.TestCase):
    def test_short_section_is_rewritten_once_before_atomic_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            config.generation.enable_story_outline = False
            config.generation.enable_story_memory_rag = False
            config.generation.enable_consumed_beat_ledger = False
            config.generation.enable_repetition_retry = False
            config.generation.enable_stability_retry = True
            config.generation.target_novel_chars = 1000
            config.generation.turn_target_chars = 1000
            config.generation.section_count = 2
            config.generation.longform_max_sections = 2
            config.generation.turn_max_sections = 2
            config.generation.section_min_chars = 600
            client = FakeClient()
            plan = {
                "mode": "JEPA",
                "direction": "서윤이 기록을 따라 다음 공간으로 이동한다",
                "examples": [],
                "beat_card": "one causal scene",
                "retrieval_mean_score": 0.7,
            }

            with patch(
                "src.generation.hallucination.plan_jepa_generation",
                return_value=plan,
            ):
                result = generate_with_controlled_hallucination(
                    config,
                    client,  # type: ignore[arg-type]
                    "기억이 거래되는 도시",
                    "서윤: 기록 복원가",
                    "서윤은 잠긴 기록실 앞에 섰다.",
                    return_details=True,
                    turn_target_chars=1000,
                )

            self.assertIsInstance(result, dict)
            assert isinstance(result, dict)
            self.assertEqual(client.calls, 2)
            self.assertIn("이어진 선택", result["text"])
            self.assertNotIn("끊긴 초고", result["text"])
            self.assertEqual(result["planner"]["turn_stability_retries"], 1)
            self.assertEqual(result["planner"]["turn_stability_retry_successes"], 1)
            draft = Path(temporary) / config.generation.longform_checkpoint_path
            self.assertTrue(draft.exists())
            self.assertIn("이어진 선택", draft.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
