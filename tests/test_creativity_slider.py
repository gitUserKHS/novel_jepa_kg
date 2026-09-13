from __future__ import annotations

import sqlite3
import tempfile
import unittest

from src.generation.longform import _creativity_temperature
from src.service.consumer_store import (
    CREATIVITY_LEVELS,
    ConsumerStore,
    creativity_bucket,
    creativity_from_profile,
    job_creativity,
)
from src.utils.config import AppConfig


class CreativityValueTests(unittest.TestCase):
    def test_numbers_strings_and_legacy_names_all_resolve(self) -> None:
        self.assertEqual(creativity_from_profile(0.8), 0.8)
        self.assertEqual(creativity_from_profile("0.25"), 0.25)
        self.assertEqual(creativity_from_profile(1), 1.0)
        self.assertEqual(creativity_from_profile("bold"), CREATIVITY_LEVELS["bold"])

    def test_out_of_range_and_garbage_are_rejected(self) -> None:
        for bad in (-0.1, 1.5, "wild", None, float("nan")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                creativity_from_profile(bad)

    def test_bucket_is_the_nearest_legacy_name(self) -> None:
        self.assertEqual(creativity_bucket(0.0), "stable")
        self.assertEqual(creativity_bucket(0.35), "balanced")
        self.assertEqual(creativity_bucket(0.9), "bold")

    def test_job_rows_without_the_new_column_fall_back_to_the_name(self) -> None:
        self.assertEqual(job_creativity({"creativity": 0.7, "creativity_profile": "stable"}), 0.7)
        self.assertEqual(job_creativity({"creativity": None, "creativity_profile": "stable"}), 0.2)
        self.assertEqual(job_creativity({"creativity_profile": "bold"}), 0.5)


class TemperatureMappingTests(unittest.TestCase):
    def _temperature(self, level: float) -> float:
        config = AppConfig()
        config.generation.hallucination_target = level
        return _creativity_temperature(config)

    def test_temperature_rises_monotonically_across_the_slider(self) -> None:
        points = [self._temperature(i / 20) for i in range(21)]
        for lower, higher in zip(points, points[1:]):
            self.assertLessEqual(lower, higher)
        self.assertLess(points[0], points[-1])

    def test_the_legacy_presets_keep_their_measured_temperatures(self) -> None:
        base = AppConfig().llm.novel_temperature
        self.assertAlmostEqual(self._temperature(0.35), base, places=6)
        self.assertAlmostEqual(self._temperature(0.20), base - 0.09, places=6)
        self.assertAlmostEqual(self._temperature(0.50), base + 0.09, places=6)

    def test_the_ends_stay_inside_the_sampling_bounds(self) -> None:
        self.assertGreaterEqual(self._temperature(0.0), 0.3)
        self.assertLessEqual(self._temperature(1.0), 1.1)


class SliderStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.user = self.store.create_user(username="dial.user", display_name="Dial", password="dial-password-1")
        self.uid = str(self.user["id"])
        story = self.store.create_story(
            self.uid, title="강도 실험", genre="SF", premise="p", world="w", protagonist="x: y",
            target_chars=10000, research_consent=False,
        )
        self.sid = str(story["id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _job(self, job_id: int) -> dict:
        with self.store.connect() as connection:
            return dict(connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def test_a_slider_value_is_stored_exactly_and_bucketed_for_legacy_readers(self) -> None:
        job = self.store.enqueue_job(self.uid, self.sid, instruction="첫 장면", creativity=0.8, requested_chars=2000)
        row = self._job(int(job["id"]))
        self.assertAlmostEqual(float(row["creativity"]), 0.8)
        self.assertEqual(row["creativity_profile"], "bold")
        self.assertAlmostEqual(job_creativity(row), 0.8)

    def test_legacy_callers_can_still_pass_a_profile_name(self) -> None:
        job = self.store.enqueue_job(self.uid, self.sid, instruction="첫 장면", creativity_profile="stable",
                                     requested_chars=2000)
        row = self._job(int(job["id"]))
        self.assertAlmostEqual(float(row["creativity"]), 0.2)
        self.assertEqual(row["creativity_profile"], "stable")

    def test_values_outside_zero_to_one_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.store.enqueue_job(self.uid, self.sid, instruction="첫 장면", creativity=1.5, requested_chars=2000)
        with self.assertRaises(ValueError):
            self.store.enqueue_job(self.uid, self.sid, instruction="첫 장면", requested_chars=2000)

    def test_section_metrics_carry_the_exact_value(self) -> None:
        job = self.store.enqueue_job(self.uid, self.sid, instruction="첫 장면", creativity=0.05, requested_chars=2000)
        self.store.save_section_metrics(story_id=self.sid, job_id=int(job["id"]), model_version="v1",
                                        creativity=0.05, values=[(1, {"repetition_rate": 0.1})])
        rows = self.store.anonymous_metric_rows()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["creativity"]), 0.05)
        self.assertEqual(rows[0]["creativity_profile"], "stable")


class MigrationTests(unittest.TestCase):
    """A database from before the slider only has the three names; opening it must add and backfill the value."""

    def test_an_old_database_gains_the_column_and_a_backfilled_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            store = ConsumerStore(config)
            user = store.create_user(username="old.user", display_name="Old", password="old-password-1")
            story = store.create_story(str(user["id"]), title="옛 작품", genre="SF", premise="p", world="w",
                                       protagonist="x: y", target_chars=10000, research_consent=False)
            job = store.enqueue_job(str(user["id"]), str(story["id"]), instruction="첫 장면",
                                    creativity_profile="bold", requested_chars=2000)
            store.save_section_metrics(story_id=str(story["id"]), job_id=int(job["id"]), model_version="v1",
                                       creativity_profile="bold", values=[(1, {})])
            # Turn it back into a pre-slider database: drop the value columns, keep the names.
            connection = sqlite3.connect(store.database_path)
            connection.execute("ALTER TABLE jobs DROP COLUMN creativity")
            connection.execute("ALTER TABLE section_metrics DROP COLUMN creativity")
            connection.commit()
            connection.close()

            reopened = ConsumerStore(config)

            with reopened.connect() as conn:
                job_row = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job["id"]),)).fetchone())
                metric_row = dict(conn.execute("SELECT * FROM section_metrics").fetchone())
            self.assertAlmostEqual(float(job_row["creativity"]), CREATIVITY_LEVELS["bold"])
            self.assertAlmostEqual(float(metric_row["creativity"]), CREATIVITY_LEVELS["bold"])
            self.assertAlmostEqual(job_creativity(job_row), CREATIVITY_LEVELS["bold"])


if __name__ == "__main__":
    unittest.main()
