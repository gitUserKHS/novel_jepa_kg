from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generation.hallucination import generate_with_controlled_hallucination
from src.utils.config import AppConfig


PLAN = {
    "mode": "JEPA",
    "direction": "서윤이 기록의 끝을 향해 이동한다",
    "examples": [],
    "beat_card": "one causal scene",
    "retrieval_mean_score": 0.7,
}


class OvershootClient:
    """Returns sections much longer than the per-section projection."""

    dry_run = False

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        body = (
            "서윤은 단서의 순서를 바로잡고 잠긴 기록실의 문을 열었다. "
            "그 선택의 대가로 조명이 하나씩 꺼졌고, 다음 방으로 향하는 길이 드러났다. "
        ) * 22
        return f"### 장면 {self.calls}\n\n{body.strip()}"


def ending_test_config(temporary: str) -> AppConfig:
    config = AppConfig(output_root=temporary)
    config.generation.enable_story_outline = False
    config.generation.enable_story_memory_rag = False
    config.generation.enable_consumed_beat_ledger = False
    config.generation.enable_repetition_retry = False
    config.generation.enable_stability_retry = False
    config.generation.enable_consistency_repair = False
    config.generation.target_novel_chars = 1500
    config.generation.turn_target_chars = 1200
    config.generation.section_count = 3
    config.generation.section_min_chars = 600
    config.generation.turn_max_sections = 4
    config.generation.longform_max_sections = 8
    return config


class LongformEndingContractTests(unittest.TestCase):
    def test_turn_that_overshoots_target_still_lands_the_finale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = ending_test_config(temporary)
            client = OvershootClient()

            with patch(
                "src.generation.hallucination.plan_jepa_generation",
                return_value=PLAN,
            ):
                result = generate_with_controlled_hallucination(
                    config,
                    client,  # type: ignore[arg-type]
                    "기억이 거래되는 도시",
                    "서윤: 기록 복원가",
                    "서윤은 잠긴 기록실 앞에 섰다.",
                    return_details=True,
                    turn_target_chars=1200,
                )

            self.assertIsInstance(result, dict)
            assert isinstance(result, dict)
            # Section 1 overshoots past the whole target, then the loop must
            # still append the closing section instead of stranding the draft.
            self.assertEqual(client.calls, 2)
            self.assertEqual(result["planner"]["completed_sections"], 2)
            self.assertTrue(result["planner"]["novel_completed"])
            draft = Path(temporary) / config.generation.longform_checkpoint_path
            self.assertIn("장면 1", draft.read_text(encoding="utf-8"))
            self.assertIn("장면 2", draft.read_text(encoding="utf-8"))
            state_path = Path(temporary) / config.generation.longform_state_path
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(state["novel_completed"])

    def test_completed_novel_refuses_further_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = ending_test_config(temporary)
            first_client = OvershootClient()
            with patch(
                "src.generation.hallucination.plan_jepa_generation",
                return_value=PLAN,
            ):
                generate_with_controlled_hallucination(
                    config,
                    first_client,  # type: ignore[arg-type]
                    "기억이 거래되는 도시",
                    "서윤: 기록 복원가",
                    "서윤은 잠긴 기록실 앞에 섰다.",
                    return_details=True,
                    turn_target_chars=1200,
                )

                second_client = OvershootClient()
                with self.assertRaises(RuntimeError) as raised:
                    generate_with_controlled_hallucination(
                        config,
                        second_client,  # type: ignore[arg-type]
                        "기억이 거래되는 도시",
                        "서윤: 기록 복원가",
                        "서윤은 잠긴 기록실 앞에 섰다.",
                        return_details=True,
                        continue_existing=True,
                        turn_target_chars=1200,
                    )

            self.assertIn("결말", str(raised.exception))
            self.assertEqual(second_client.calls, 0)


if __name__ == "__main__":
    unittest.main()
