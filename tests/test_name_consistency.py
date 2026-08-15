from __future__ import annotations

import unittest

from src.generation.consistency import check_name_consistency, extract_character_names
from src.generation.stability import assess_section_stability
from src.memory.story_rag import StoryMemory


CHARACTERS = "에블린: 시한부 판정을 받은 공작가 영애\n칼릭스: 황태자\n레이: 호위 기사"


class HonorificFalsePositiveTests(unittest.TestCase):
    """Role address must not read as an invented character name.

    The generation prompt tells the model to use role words instead of new
    proper nouns; the checker flagging those same words made the stability
    gate rewrite sections for following instructions.
    """

    def test_role_address_is_not_an_unknown_name(self) -> None:
        prose = (
            "### 연회\n\n"
            "\"아가씨, 이러시면 안 됩니다.\" 하인이 말렸지만 에블린은 걸음을 멈추지 않았다. "
            "도련님 소리를 듣던 시절의 칼릭스라면 상상도 못 했을 표정이었다. "
            "스승님의 가르침도, 선생님의 경고도, 공작님의 명령도 지금은 멀었다."
        )

        check = check_name_consistency(prose, CHARACTERS)

        self.assertEqual(check.issues, [])
        self.assertEqual(check.score, 1.0)

    def test_a_known_name_with_an_honorific_is_still_fine(self) -> None:
        check = check_name_consistency("에블린님께서 먼저 입을 여셨다.", CHARACTERS)

        self.assertEqual(check.issues, [])

    def test_extraction_still_reads_the_character_sheet(self) -> None:
        self.assertEqual(extract_character_names(CHARACTERS), ["에블린", "칼릭스", "레이"])

    def test_stability_gate_passes_a_section_full_of_role_address(self) -> None:
        section = (
            "### 밤의 정원\n\n"
            + ("\"아가씨, 부디 안으로 드시지요.\" 시녀장이 청했지만 에블린은 도련님과의 "
               "약속을 떠올리며 정원 끝까지 걸었다. 선생님이라면 뭐라고 했을까. ") * 12
            + "그날의 결심이 그녀를 여기까지 데려온 것이었다."
        )

        assessment = assess_section_stability(
            section,
            StoryMemory(section_index=1, summary="에블린이 정원에서 결심을 굳힌다"),
            ledger={},
            characters=CHARACTERS,
            prior_titles=[],
            minimum_chars=300,
        )

        self.assertEqual(assessment.issues, [])
        self.assertFalse(assessment.hard_failure)

    def test_a_truncated_section_is_still_caught(self) -> None:
        section = "### 연회\n\n" + ("에블린은 홀을 가로질러 걸었다. " * 30) + "그리고 그녀는"

        assessment = assess_section_stability(
            section,
            StoryMemory(section_index=1, summary="이동"),
            ledger={},
            characters=CHARACTERS,
            prior_titles=[],
            minimum_chars=300,
        )

        self.assertIn("section appears to end mid-sentence", assessment.issues)
        self.assertTrue(assessment.hard_failure)


if __name__ == "__main__":
    unittest.main()
