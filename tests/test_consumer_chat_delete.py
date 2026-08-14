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
from src.service.story_workspace import StoryWorkspace
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

    def _workspace(self) -> StoryWorkspace:
        return StoryWorkspace.for_story(self.config, str(self.story["id"]), create=True)

    def _write_manuscript(self) -> StoryWorkspace:
        workspace = self._workspace()
        workspace.draft.write_text("### 1장\n\n에블린은 파혼 서류를 던졌다.", encoding="utf-8")
        workspace.memory.write_text('{"section_index": 1}\n', encoding="utf-8")
        workspace.ledger.write_text('{"current_states": []}', encoding="utf-8")
        workspace.outline.write_text('{"beats": []}', encoding="utf-8")
        workspace.state.write_text('{"turn": 1}', encoding="utf-8")
        workspace.live.write_text("집필 중이던 조각", encoding="utf-8")
        return workspace

    def test_reset_clears_the_manuscript_and_everything_derived_from_it(self) -> None:
        workspace = self._write_manuscript()
        self._finished_job()

        self.store.reset_owned_story(str(self.owner["id"]), str(self.story["id"]))

        for path in (
            workspace.draft,
            workspace.memory,
            workspace.ledger,
            workspace.outline,
            workspace.state,
            workspace.live,
        ):
            self.assertFalse(path.exists(), f"{path.name} survived the reset")
        self.assertEqual(self._owner_job_ids(), [])

    def test_reset_zeroes_progress_but_keeps_the_settings(self) -> None:
        self._write_manuscript()
        self._finished_job()

        self.store.reset_owned_story(str(self.owner["id"]), str(self.story["id"]))

        story = self.store.get_owned_story(str(self.owner["id"]), str(self.story["id"]))
        self.assertIsNotNone(story)
        self.assertEqual(int(story["current_chars"]), 0)
        self.assertEqual(int(story["section_count"]), 0)
        self.assertIsNone(story["completed_at"])
        self.assertEqual(story["title"], "파혼 선언")
        self.assertEqual(story["genre"], "궁중 로맨스 판타지")
        self.assertIn("에블린", str(story["protagonist"]))

    def test_reset_is_refused_while_a_turn_is_still_queued(self) -> None:
        workspace = self._write_manuscript()
        self.store.enqueue_job(
            str(self.owner["id"]),
            str(self.story["id"]),
            instruction="아직 진행 중.",
            requested_chars=2000,
            creativity_profile="balanced",
        )

        with self.assertRaises(ConsumerStoreError):
            self.store.reset_owned_story(str(self.owner["id"]), str(self.story["id"]))

        self.assertTrue(workspace.draft.exists())

    def test_another_account_cannot_reset(self) -> None:
        workspace = self._write_manuscript()

        with self.assertRaises(AuthorizationError):
            self.store.reset_owned_story(str(self.intruder["id"]), str(self.story["id"]))

        self.assertTrue(workspace.draft.exists())

    def test_delete_removes_the_row_from_the_database(self) -> None:
        self._write_manuscript()
        self._finished_job()
        story_id = str(self.story["id"])

        self.store.delete_owned_story(str(self.owner["id"]), story_id)

        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT COUNT(*) AS total FROM stories WHERE id = ?", (story_id,)
            ).fetchone()
            jobs = connection.execute(
                "SELECT COUNT(*) AS total FROM jobs WHERE story_id = ?", (story_id,)
            ).fetchone()
        self.assertEqual(int(rows["total"]), 0)
        self.assertEqual(int(jobs["total"]), 0, "jobs should cascade with the story")

    def test_delete_removes_the_story_files(self) -> None:
        workspace = self._write_manuscript()

        self.store.delete_owned_story(str(self.owner["id"]), str(self.story["id"]))

        self.assertFalse(workspace.root.exists())

    def test_another_account_cannot_delete_the_story(self) -> None:
        with self.assertRaises(AuthorizationError):
            self.store.delete_owned_story(str(self.intruder["id"]), str(self.story["id"]))

        self.assertIsNotNone(
            self.store.get_owned_story(str(self.owner["id"]), str(self.story["id"]))
        )


if __name__ == "__main__":
    unittest.main()
