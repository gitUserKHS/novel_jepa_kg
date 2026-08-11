from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.generation.coherence import (
    assess_section_coherence,
    realized_state_text,
)
from src.generation.hallucination import generate_with_controlled_hallucination
from src.memory.beat_ledger import (
    BEAT_TRIGGERS,
    GENRE_BEAT_TRIGGERS,
    extract_consumed_beats,
    triggers_for_genre,
)
from src.memory.story_rag import StateUpdate, StoryMemory
from src.utils.config import AppConfig


PALACE_SECTION = (
    "### 밀지\n\n"
    "왕은 밀지를 내려 역모에 얽힌 이름을 지웠다. "
    "편전 밖에서는 자객이 숨을 죽였고, 입궐한 신하들은 서로의 충성을 저울질했다."
)


class GenreTriggerTests(unittest.TestCase):
    def test_every_genre_table_uses_the_known_beat_types(self) -> None:
        for genre, table in GENRE_BEAT_TRIGGERS.items():
            with self.subTest(genre=genre):
                self.assertEqual(set(table), set(BEAT_TRIGGERS))
                for beat_type, words in table.items():
                    self.assertTrue(words, f"{genre}/{beat_type} has no triggers")

    def test_an_unknown_genre_falls_back_to_the_base_table(self) -> None:
        self.assertIs(triggers_for_genre("존재하지 않는 장르"), BEAT_TRIGGERS)
        self.assertIs(triggers_for_genre(None), BEAT_TRIGGERS)

    def test_genre_triggers_extend_rather_than_replace_the_base(self) -> None:
        table = triggers_for_genre("궁중 판타지")

        for beat_type, base_words in BEAT_TRIGGERS.items():
            self.assertTrue(set(base_words).issubset(set(table[beat_type])))
        self.assertIn("밀지", table["reveal"])

    def test_court_prose_is_only_detected_with_its_genre_table(self) -> None:
        base = extract_consumed_beats(PALACE_SECTION, section_index=1)
        court = extract_consumed_beats(PALACE_SECTION, section_index=1, genre="궁중 판타지")

        self.assertEqual(base, [])
        self.assertTrue({beat.beat_type for beat in court} >= {"reveal", "new_threat"})


class RealizedStateTextTests(unittest.TestCase):
    def test_it_mirrors_the_training_target_shape(self) -> None:
        memory = StoryMemory(
            section_index=2,
            summary="서윤이 기록실을 열었다",
            keywords=["불안", "확신"],
            open_clues=["붉은 열쇠의 주인"],
            state_updates=[StateUpdate(entity="서윤", attribute="location", value="기록실")],
        )

        text = realized_state_text(memory, "### 장면\n\n본문")

        self.assertTrue(text.startswith("[Target Encoder Input]"))
        for label in ("다음 장면 요약", "다음 감정", "다음 갈등", "다음 상태", "다음 훅"):
            self.assertIn(label, text)
        self.assertIn("서윤이 기록실을 열었다", text)

    def test_an_empty_memory_falls_back_to_the_section_body(self) -> None:
        section = "### 제목\n\n서윤은 문을 열고 복도로 나섰다."

        text = realized_state_text(StoryMemory(section_index=1), section)

        self.assertIn("서윤은 문을 열고", text)
        self.assertNotIn("### 제목", text)


class StubClient:
    dry_run = False

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.embed_calls = 0

    def embed(self, texts: list[str], *, unload_chat: bool | None = None) -> np.ndarray:
        self.embed_calls += 1
        if self.vectors is None:
            raise RuntimeError("embedding backend is down")
        return np.asarray(self.vectors[: len(texts)], dtype="float32")


class CoherenceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = StoryMemory(section_index=1, summary="서윤이 기록실을 열었다")
        self.section = "### 장면\n\n서윤은 문을 열었다."

    def assess(self, client: StubClient, config: AppConfig):
        return assess_section_coherence(
            config,
            client,  # type: ignore[arg-type]
            world="기억이 거래되는 도시",
            characters="서윤: 기록 복원가",
            preceding_context="서윤은 잠긴 기록실 앞에 섰다.",
            section=self.section,
            memory=self.memory,
        )

    def test_an_aligned_section_scores_high_and_raises_no_issue(self) -> None:
        client = StubClient([[1.0, 0.0], [1.0, 0.0]])
        with patch(
            "src.generation.coherence.predict_from_context_embeddings",
            return_value=np.asarray([[1.0, 0.0]], dtype="float32"),
        ):
            result = self.assess(client, AppConfig())

        self.assertTrue(result.available)
        self.assertAlmostEqual(result.score, 1.0, places=4)
        self.assertEqual(result.issues, [])
        self.assertEqual(client.embed_calls, 1)

    def test_a_non_sequitur_section_is_flagged(self) -> None:
        client = StubClient([[1.0, 0.0], [0.0, 1.0]])
        with patch(
            "src.generation.coherence.predict_from_context_embeddings",
            return_value=np.asarray([[1.0, 0.0]], dtype="float32"),
        ):
            result = self.assess(client, AppConfig())

        self.assertTrue(result.available)
        self.assertAlmostEqual(result.score, 0.0, places=4)
        self.assertEqual(len(result.issues), 1)
        self.assertIn("JEPA plausibility", result.issues[0])

    def test_a_backend_failure_downgrades_instead_of_raising(self) -> None:
        result = self.assess(StubClient(None), AppConfig())

        self.assertFalse(result.available)
        self.assertEqual(result.issues, [])

    def test_the_gate_can_be_disabled(self) -> None:
        config = AppConfig()
        config.generation.enable_jepa_coherence_gate = False
        client = StubClient([[1.0, 0.0], [0.0, 1.0]])

        result = self.assess(client, config)

        self.assertFalse(result.available)
        self.assertEqual(client.embed_calls, 0)

    def test_the_default_threshold_matches_the_calibration(self) -> None:
        # Measured 2026-08-06 against artifact 20260806T111529Z-7262bfe2 over
        # two genres: genuine sections 0.651-0.766, deliberate causal breaks
        # 0.502-0.637. The default must separate them cleanly.
        threshold = AppConfig().generation.jepa_coherence_min_cosine
        lowest_genuine = 0.651
        highest_break = 0.637

        self.assertLess(threshold, lowest_genuine)
        self.assertGreater(threshold, highest_break)

    def test_the_gate_keeps_the_chat_model_resident_by_default(self) -> None:
        recorded: dict[str, object] = {}

        class RecordingClient(StubClient):
            def embed(self, texts, *, unload_chat=None):
                recorded["unload_chat"] = unload_chat
                return super().embed(texts)

        with patch(
            "src.generation.coherence.predict_from_context_embeddings",
            return_value=np.asarray([[1.0, 0.0]], dtype="float32"),
        ):
            self.assess(RecordingClient([[1.0, 0.0], [1.0, 0.0]]), AppConfig())

        # Evicting the chat model costs more in the next reload than the embed.
        self.assertIs(recorded["unload_chat"], False)


class PrimaryFunctionRotationTests(unittest.TestCase):
    def test_different_stories_start_the_cycle_at_different_points(self) -> None:
        from src.memory.beat_ledger import PRIMARY_FUNCTIONS, choose_primary_function

        first = [choose_primary_function(i, [], story_seed="궁중 판타지\n왕궁")[0] for i in range(1, 7)]
        second = [
            choose_primary_function(i, [], story_seed="해양 모험\n항구")[0] for i in range(1, 7)
        ]

        self.assertNotEqual(first, second)
        # Both still cover every function once per cycle.
        self.assertEqual(set(first), {name for name, _rule in PRIMARY_FUNCTIONS})
        self.assertEqual(set(second), {name for name, _rule in PRIMARY_FUNCTIONS})

    def test_the_same_story_keeps_its_phase_across_turns(self) -> None:
        from src.memory.beat_ledger import choose_primary_function

        seed = "한국형 SF 미스터리\n기억이 거래되는 도시"
        first_pass = [choose_primary_function(i, [], story_seed=seed)[0] for i in range(1, 9)]
        second_pass = [choose_primary_function(i, [], story_seed=seed)[0] for i in range(1, 9)]

        self.assertEqual(first_pass, second_pass)

    def test_an_empty_seed_keeps_the_original_rotation(self) -> None:
        from src.memory.beat_ledger import choose_primary_function

        self.assertEqual(choose_primary_function(1, [])[0], "new_clue")


class LongformClient:
    """First section is a non-sequitur; the rewrite lands back on the direction."""

    dry_run = False

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        body = (
            "서윤은 기록의 순서를 맞춘 뒤 잠긴 문을 열었다. "
            "그 선택의 대가로 복도의 조명이 하나씩 꺼졌다. "
        ) * 18
        title = "이어진 선택" if self.calls > 1 else "빗나간 장면"
        return f"### {title}\n\n{body.strip()}"

    def embed(self, texts: list[str], *, unload_chat: bool | None = None) -> np.ndarray:
        # Context vector stays fixed; the realized state only aligns after the
        # rewrite, so the gate must fire once and then clear.
        realized = [1.0, 0.0] if self.calls > 1 else [0.0, 1.0]
        return np.asarray([[1.0, 0.0], realized][: len(texts)], dtype="float32")


class CoherenceRetryIntegrationTests(unittest.TestCase):
    def test_a_low_coherence_section_is_rewritten_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            config.generation.enable_story_outline = False
            config.generation.enable_story_memory_rag = False
            config.generation.enable_consumed_beat_ledger = False
            config.generation.enable_repetition_retry = False
            config.generation.enable_stability_retry = False
            config.generation.enable_jepa_coherence_gate = True
            config.generation.target_novel_chars = 5000
            config.generation.turn_target_chars = 1000
            config.generation.section_min_chars = 600
            config.generation.turn_max_sections = 2
            config.generation.longform_max_sections = 2
            client = LongformClient()
            plan = {
                "mode": "JEPA",
                "direction": "서윤이 기록을 따라 이동한다",
                "examples": [],
                "beat_card": "one causal scene",
                "retrieval_mean_score": 0.7,
            }

            with patch(
                "src.generation.hallucination.plan_jepa_generation",
                return_value=plan,
            ), patch(
                "src.generation.coherence.predict_from_context_embeddings",
                return_value=np.asarray([[1.0, 0.0]], dtype="float32"),
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

            assert isinstance(result, dict)
            planner = result["planner"]
            self.assertEqual(planner["turn_coherence_retries"], 1)
            self.assertEqual(planner["turn_coherence_retry_successes"], 1)
            self.assertIn("이어진 선택", result["text"])
            self.assertNotIn("빗나간 장면", result["text"])
            self.assertAlmostEqual(planner["mean_jepa_coherence"], 1.0, places=4)
            # The kept score must describe the section that survived, and it is
            # reported per section so the research rows can use it.
            self.assertEqual(planner["jepa_coherence_by_section"], {1: 1.0})

    def test_the_run_state_records_the_plausibility_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            config.generation.enable_story_outline = False
            config.generation.enable_story_memory_rag = False
            config.generation.enable_consumed_beat_ledger = False
            config.generation.enable_repetition_retry = False
            config.generation.enable_stability_retry = False
            config.generation.target_novel_chars = 5000
            config.generation.turn_target_chars = 1000
            config.generation.section_min_chars = 600
            config.generation.turn_max_sections = 1
            config.generation.longform_max_sections = 1
            client = LongformClient()
            client.calls = 1  # make the first section already aligned

            with patch(
                "src.generation.hallucination.plan_jepa_generation",
                return_value={
                    "mode": "JEPA",
                    "direction": "이동",
                    "examples": [],
                    "beat_card": "one causal scene",
                    "retrieval_mean_score": 0.7,
                },
            ), patch(
                "src.generation.coherence.predict_from_context_embeddings",
                return_value=np.asarray([[1.0, 0.0]], dtype="float32"),
            ):
                generate_with_controlled_hallucination(
                    config,
                    client,  # type: ignore[arg-type]
                    "기억이 거래되는 도시",
                    "서윤: 기록 복원가",
                    "서윤은 기록실 앞에 섰다.",
                    return_details=True,
                    turn_target_chars=1000,
                )

            state = json.loads(
                (Path(temporary) / config.generation.longform_state_path).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["jepa_coherence_scored_sections"], 1)
            self.assertAlmostEqual(state["mean_jepa_coherence"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
