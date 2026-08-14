from __future__ import annotations

import tempfile
import unittest

from src.service.consumer_store import (
    JOB_QUEUED,
    JOB_RUNNING,
    AuthorizationError,
    ConsumerStore,
    ConsumerStoreError,
)
from src.utils.config import AppConfig


class ChatDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.owner = self.store.create_user(
            username="evelyn.writer",
            display_name="Evelyn",
            password="correct-horse-1",
        )
        self.intruder = self.store.create_user(
            username="calix.reader",
            display_name="Calix",
            password="correct-horse-2",
        )
        self.story = self.store.create_story(
            str(self.owner["id"]),
            title="파혼 선언",
            genre="궁중 로맨스 판타지",
            premise="시한부 판정을 받은 악녀가 파혼을 선언한다.",
            world="황실과 귀족 가문이 예언으로 얽힌 제국",
            protagonist="에블린: 시한부 판정을 받은 공작가 영애",
            characters="칼릭스: 황태자",
            target_chars=10000,
            research_consent=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _finished_job(self) -> int:
        job = self.store.enqueue_job(
            str(self.owner["id"]),
            str(self.story["id"]),
            instruction="첫 장면을 써줘.",
            requested_chars=2000,
            creativity_profile="balanced",
        )
        job_id = int(job["id"])
        self.store.claim_next_job("worker-1", "v1")
        self.store.complete_job(
            job_id,
            result_chars=1800,
            result_section_count=1,
            total_chars=1800,
            total_section_count=1,
            metrics={},
        )
        return job_id

    def _owner_job_ids(self) -> list[int]:
        return [
            int(row["id"])
            for row in self.store.list_owned_jobs(str(self.owner["id"]), str(self.story["id"]))
        ]

    def test_a_finished_turn_can_be_deleted(self) -> None:
        job_id = self._finished_job()

        self.store.delete_owned_job(str(self.owner["id"]), str(self.story["id"]), job_id)

        self.assertNotIn(job_id, self._owner_job_ids())

    def test_deleting_a_turn_keeps_the_story_and_its_progress(self) -> None:
        job_id = self._finished_job()

        self.store.delete_owned_job(str(self.owner["id"]), str(self.story["id"]), job_id)

        story = self.store.get_owned_story(str(self.owner["id"]), str(self.story["id"]))
        self.assertIsNotNone(story)
        self.assertEqual(int(story["current_chars"]), 1800)

    def test_another_account_cannot_delete_a_turn(self) -> None:
        job_id = self._finished_job()

        with self.assertRaises(AuthorizationError):
            self.store.delete_owned_job(str(self.intruder["id"]), str(self.story["id"]), job_id)

        self.assertIn(job_id, self._owner_job_ids())

    def test_a_queued_turn_cannot_be_deleted(self) -> None:
        job = self.store.enqueue_job(
            str(self.owner["id"]),
            str(self.story["id"]),
            instruction="다음 장면.",
            requested_chars=2000,
            creativity_profile="balanced",
        )

        with self.assertRaises(ConsumerStoreError):
            self.store.delete_owned_job(
                str(self.owner["id"]), str(self.story["id"]), int(job["id"])
            )

        self.assertIn(int(job["id"]), self._owner_job_ids())

    def test_a_running_turn_cannot_be_deleted(self) -> None:
        job = self.store.enqueue_job(
            str(self.owner["id"]),
            str(self.story["id"]),
            instruction="다음 장면.",
            requested_chars=2000,
            creativity_profile="balanced",
        )
        self.store.claim_next_job("worker-1", "v1")

        with self.assertRaises(ConsumerStoreError):
            self.store.delete_owned_job(
                str(self.owner["id"]), str(self.story["id"]), int(job["id"])
            )

    def test_clearing_history_removes_finished_turns_only(self) -> None:
        first = self._finished_job()
        second = self._finished_job()
        active = self.store.enqueue_job(
            str(self.owner["id"]),
            str(self.story["id"]),
            instruction="아직 진행 중.",
            requested_chars=2000,
            creativity_profile="balanced",
        )

        removed = self.store.clear_owned_job_history(str(self.owner["id"]), str(self.story["id"]))

        self.assertEqual(removed, 2)
        remaining = self._owner_job_ids()
        self.assertEqual(remaining, [int(active["id"])])
        self.assertNotIn(first, remaining)
        self.assertNotIn(second, remaining)

    def test_another_account_cannot_clear_history(self) -> None:
        self._finished_job()

        with self.assertRaises(AuthorizationError):
            self.store.clear_owned_job_history(str(self.intruder["id"]), str(self.story["id"]))

        self.assertEqual(len(self._owner_job_ids()), 1)

    def test_clearing_an_empty_history_is_a_no_op(self) -> None:
        removed = self.store.clear_owned_job_history(str(self.owner["id"]), str(self.story["id"]))

        self.assertEqual(removed, 0)

    def test_outstanding_statuses_are_the_protected_ones(self) -> None:
        """The UI hides its delete control for exactly these statuses."""
        from src.service.consumer_store import OUTSTANDING_JOB_STATUSES

        self.assertEqual(set(OUTSTANDING_JOB_STATUSES), {JOB_QUEUED, JOB_RUNNING})


if __name__ == "__main__":
    unittest.main()
