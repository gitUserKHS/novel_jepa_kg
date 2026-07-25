from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generation.hallucination import _completion_rule
from src.generation.stability import assess_section_stability
from src.memory.story_outline import (
    fallback_story_outline,
    load_story_outline,
    outline_context,
    write_story_outline,
)
from src.memory.story_rag import StateUpdate, StoryMemory


def long_section(title: str = "첫 선택") -> str:
    body = (
        "지우는 낡은 기록을 다시 펼쳐 단서의 순서를 비교했다. "
        "창문 너머의 빛이 책상 위를 지나가는 동안 그는 서두르지 않고, "
        "앞선 선택이 남긴 결과를 확인한 뒤 다음 행동을 결정했다. "
    ) * 4
    return f"### {title}\n\n{body.strip()}"


class StoryOutlineTests(unittest.TestCase):
    def test_outline_maps_opening_and_ending_sections(self) -> None:
        outline = fallback_story_outline("기억을 되찾는 선택", beat_count=8)
        opening, opening_index = outline_context(outline, section_index=1, planned_sections=8)
        ending, ending_index = outline_context(outline, section_index=8, planned_sections=8)

        self.assertEqual(len(outline.beats), 8)
        self.assertEqual(opening_index, 0)
        self.assertEqual(ending_index, 7)
        self.assertIn("active beat 1/8", opening)
        self.assertIn("active beat 8/8", ending)

    def test_outline_round_trip_is_atomic_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "outline.json"
            original = fallback_story_outline("닫힌 도시의 비밀", beat_count=6)
            write_story_outline(path, original)
            loaded = load_story_outline(path)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.model_dump(), original.model_dump())
            self.assertFalse(path.with_name(".outline.json.tmp").exists())

    def test_only_the_target_crossing_section_closes_the_novel(self) -> None:
        turn_rule, turn_is_final = _completion_rule(
            12000,
            30000,
            1800,
            final_section_of_turn=True,
        )
        final_rule, final_is_final = _completion_rule(
            28500,
            30000,
            1800,
            final_section_of_turn=False,
        )

        self.assertFalse(turn_is_final)
        self.assertIn("continuation hook", turn_rule)
        self.assertTrue(final_is_final)
        self.assertIn("final section of the novel", final_rule)
        self.assertIn("Do not introduce a new central mystery", final_rule)


class StabilityAssessmentTests(unittest.TestCase):
    def test_short_or_truncated_section_is_hard_failure(self) -> None:
        result = assess_section_stability(
            "### 중단\n\n문장이 여기서",
            StoryMemory(section_index=1),
            ledger={},
            characters="",
            prior_titles=[],
            minimum_chars=300,
        )

        self.assertTrue(result.hard_failure)
        self.assertGreaterEqual(len(result.issues), 2)

    def test_state_change_requires_an_explicit_cause(self) -> None:
        memory = StoryMemory(
            section_index=2,
            state_updates=[StateUpdate(entity="지우", attribute="location", value="부산")],
        )
        ledger = {
            "current_states": [
                {"entity": "지우", "attribute": "location", "value": "서울"}
            ]
        }
        result = assess_section_stability(
            long_section(),
            memory,
            ledger=ledger,
            characters="지우: 기록 복원가",
            prior_titles=[],
            minimum_chars=200,
        )

        self.assertTrue(any("state changed without" in issue for issue in result.issues))

    def test_resolved_clue_cannot_silently_reopen(self) -> None:
        memory = StoryMemory(section_index=3, open_clues=["붉은 열쇠의 주인"])
        ledger = {"resolved_clues": [{"text": "붉은 열쇠의 주인", "section_index": 2}]}
        result = assess_section_stability(
            long_section(),
            memory,
            ledger=ledger,
            characters="",
            prior_titles=[],
            minimum_chars=200,
        )

        self.assertTrue(any("resolved clue" in issue for issue in result.issues))

    def test_well_formed_section_passes_without_hard_failure(self) -> None:
        result = assess_section_stability(
            long_section(),
            StoryMemory(section_index=1),
            ledger={},
            characters="지우: 기록 복원가",
            prior_titles=[],
            minimum_chars=200,
        )

        self.assertFalse(result.hard_failure)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
