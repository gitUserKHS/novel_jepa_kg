from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from src.service.consumer_store import (
    JOB_FAILED_RECOVERABLE,
    AccountExistsError,
    AuthorizationError,
    ConsumerStore,
    ConsumerStoreError,
    MaintenanceModeError,
    StoryBusyError,
)
from src.service.security import split_session_token
from src.service.story_workspace import StoryWorkspace
from src.utils.config import AppConfig


class ConsumerStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.alice = self.store.create_user(
            username="alice.writer",
            display_name="Alice",
            password="correct-horse-1",
        )
        self.bob = self.store.create_user(
            username="bob_writer",
            display_name="Bob",
            password="correct-horse-2",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_story(self, owner_id: str | None = None, title: str = "Test Novel") -> dict:
        return self.store.create_story(
            owner_id or self.alice["id"],
            title=title,
            genre="SF mystery",
            premise="Track a disappearing memory",
            world="A city where memories are traded",
            protagonist="Seoyun, an archive restorer",
            target_chars=30000,
        )

    def test_account_password_and_server_session_are_hashed(self) -> None:
        self.assertNotIn("password_hash", self.alice)
        self.assertIsNone(self.store.authenticate_user("alice.writer", "wrong-password"))
        authenticated = self.store.authenticate_user("ALICE.WRITER", "correct-horse-1")
        self.assertEqual(authenticated["id"], self.alice["id"])
        with self.assertRaises(AccountExistsError):
            self.store.create_user(
                username="Alice.Writer",
                display_name="Duplicate",
                password="another-password",
            )

        token = self.store.create_user_session(self.alice["id"])
        session_id, session_secret = split_session_token(token)
        database_bytes = self.store.database_path.read_bytes().decode("latin1")
        self.assertNotIn("correct-horse-1", database_bytes)
        self.assertNotIn(session_secret, database_bytes)
        self.assertEqual(self.store.authenticate_user_session(token)["id"], self.alice["id"])
        self.assertIsNone(self.store.authenticate_user_session(f"{session_id}.wrong-secret"))
        self.store.revoke_user_session(token)
        self.assertIsNone(self.store.authenticate_user_session(token))

    def test_story_paths_and_account_ownership_are_isolated(self) -> None:
        first = self.create_story(self.alice["id"], "Alice Story")
        second = self.create_story(self.bob["id"], "Bob Story")
        self.assertEqual([story["id"] for story in self.store.list_owned_stories(self.alice["id"])], [first["id"]])
        self.assertEqual([story["id"] for story in self.store.list_owned_stories(self.bob["id"])], [second["id"]])
        self.assertIsNone(self.store.get_owned_story(self.bob["id"], first["id"]))

        first_workspace = StoryWorkspace.for_story(self.config, first["id"], create=True)
        second_workspace = StoryWorkspace.for_story(self.config, second["id"], create=True)
        first_workspace.draft.write_text("Alice draft", encoding="utf-8")
        second_workspace.draft.write_text("Bob draft", encoding="utf-8")
        self.assertNotEqual(first_workspace.root, second_workspace.root)
        self.assertEqual(second_workspace.draft.read_text(encoding="utf-8"), "Bob draft")

        with self.assertRaises(AuthorizationError):
            self.store.enqueue_job(
                self.bob["id"],
                first["id"],
                instruction="Steal the story",
                creativity_profile="balanced",
                requested_chars=3000,
            )
        with self.assertRaises(AuthorizationError):
            self.store.delete_owned_story(self.bob["id"], first["id"])
        self.assertIsNotNone(self.store.get_owned_story(self.alice["id"], first["id"]))

    def test_fifo_single_job_and_maintenance_drain(self) -> None:
        first = self.create_story(self.alice["id"], "First Story")
        second = self.create_story(self.bob["id"], "Second Story")
        first_job = self.store.enqueue_job(
            self.alice["id"],
            first["id"],
            instruction="First scene",
            creativity_profile="balanced",
            requested_chars=3000,
        )
        with self.assertRaises(StoryBusyError):
            self.store.enqueue_job(
                self.alice["id"],
                first["id"],
                instruction="Duplicate",
                creativity_profile="balanced",
                requested_chars=3000,
            )
        second_job = self.store.enqueue_job(
            self.bob["id"],
            second["id"],
            instruction="Second scene",
            creativity_profile="stable",
            requested_chars=2000,
        )
        self.assertEqual(self.store.queue_position(self.alice["id"], first_job["id"]), 1)
        self.assertEqual(self.store.queue_position(self.bob["id"], second_job["id"]), 2)
        self.assertIsNone(self.store.queue_position(self.bob["id"], first_job["id"]))

        self.store.set_maintenance(True)
        self.assertEqual(self.store.maintenance_status(), "draining")
        third = self.create_story(self.alice["id"], "Third Story")
        with self.assertRaises(MaintenanceModeError):
            self.store.enqueue_job(
                self.alice["id"],
                third["id"],
                instruction="Blocked during maintenance",
                creativity_profile="balanced",
                requested_chars=3000,
            )

        claimed = self.store.claim_next_job("worker", "v1")
        self.assertEqual(claimed["id"], first_job["id"])
        self.assertIsNone(self.store.claim_next_job("other-worker", "v1"))
        self.store.complete_job(
            first_job["id"],
            result_chars=3000,
            result_section_count=1,
            total_chars=3000,
            total_section_count=1,
            metrics={},
        )
        claimed_second = self.store.claim_next_job("worker", "v1")
        self.assertEqual(claimed_second["id"], second_job["id"])
        self.store.complete_job(
            second_job["id"],
            result_chars=2000,
            result_section_count=1,
            total_chars=2000,
            total_section_count=1,
            metrics={},
        )
        self.assertEqual(self.store.maintenance_status(), "1")
        self.store.set_maintenance(False)
        self.assertEqual(self.store.maintenance_status(), "0")

    def test_stale_worker_becomes_recoverable_and_partial_workspace_survives(self) -> None:
        story = self.create_story()
        job = self.store.enqueue_job(
            self.alice["id"],
            story["id"],
            instruction="Start",
            creativity_profile="bold",
            requested_chars=5000,
        )
        claimed = self.store.claim_next_job("dead-worker", "v1")
        self.assertEqual(claimed["id"], job["id"])
        workspace = StoryWorkspace.for_story(self.config, story["id"], create=True)
        workspace.draft.write_text("### Section 1\n\nPartial draft", encoding="utf-8")
        old = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds")
        with self.store.connect() as connection:
            connection.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (old, job["id"]))
        self.assertEqual(self.store.recover_stale_jobs(stale_after_sec=10), 1)
        recovered = self.store.get_job(job["id"])
        self.assertEqual(recovered["status"], JOB_FAILED_RECOVERABLE)
        self.assertTrue(workspace.draft.exists())

    def test_immediate_owned_delete_removes_database_row_and_workspace(self) -> None:
        story = self.create_story()
        workspace = StoryWorkspace.for_story(self.config, story["id"], create=True)
        workspace.draft.write_text("Draft", encoding="utf-8")
        self.store.delete_owned_story(self.alice["id"], story["id"])
        self.assertIsNone(self.store.get_owned_story(self.alice["id"], story["id"]))
        self.assertFalse(workspace.root.exists())

    def test_story_target_accepts_selected_value_and_enforces_range(self) -> None:
        selected = self.store.create_story(
            self.alice["id"],
            title="42K Story",
            genre="Mystery",
            premise="A sealed room opens",
            world="A city under permanent rain",
            protagonist="Jin, an investigator",
            target_chars=42000,
        )
        self.assertEqual(selected["target_chars"], 42000)

        common = {
            "owner_id": self.alice["id"],
            "title": "Invalid target",
            "genre": "Mystery",
            "premise": "A missing map",
            "world": "An underground archive",
            "protagonist": "Jin, an investigator",
        }
        with self.assertRaisesRegex(ValueError, "10,000자 이상"):
            self.store.create_story(**common, target_chars=9000)
        with self.assertRaisesRegex(ValueError, "50,000자 이하"):
            self.store.create_story(**common, target_chars=51000)

    def test_reaching_target_without_an_ending_still_allows_continuation(self) -> None:
        story = self.create_story()
        job = self.store.enqueue_job(
            self.alice["id"],
            story["id"],
            instruction="Overshoot the target",
            creativity_profile="balanced",
            requested_chars=5000,
        )
        self.store.claim_next_job("worker", "v1")
        self.store.complete_job(
            job["id"],
            result_chars=31000,
            result_section_count=8,
            total_chars=31000,
            total_section_count=8,
            metrics={"novel_completed": False},
            novel_completed=False,
        )

        reopened = self.store.get_owned_story(self.alice["id"], story["id"])
        self.assertIsNone(reopened["completed_at"])
        finale = self.store.enqueue_job(
            self.alice["id"],
            story["id"],
            instruction="Write the ending",
            creativity_profile="balanced",
            requested_chars=5000,
        )
        self.store.claim_next_job("worker", "v1")
        self.store.complete_job(
            finale["id"],
            result_chars=2000,
            result_section_count=1,
            total_chars=33000,
            total_section_count=9,
            metrics={"novel_completed": True},
            novel_completed=True,
        )

        completed = self.store.get_owned_story(self.alice["id"], story["id"])
        self.assertIsNotNone(completed["completed_at"])
        with self.assertRaisesRegex(ConsumerStoreError, "결말까지 완성"):
            self.store.enqueue_job(
                self.alice["id"],
                story["id"],
                instruction="Keep going past the ending",
                creativity_profile="balanced",
                requested_chars=2000,
            )


if __name__ == "__main__":
    unittest.main()
