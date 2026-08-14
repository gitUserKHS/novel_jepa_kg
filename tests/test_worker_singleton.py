from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.service.job_lock import ServiceBusyError, acquire_lock_file
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


class WorkerSingletonLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.path = resolve_path(self.config, self.config.consumer.worker_lock_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_the_first_worker_takes_the_lock(self) -> None:
        with acquire_lock_file(self.path, "consumer worker") as lease:
            self.assertTrue(self.path.exists())
            self.assertEqual(lease.operation, "consumer worker")

    def test_a_second_worker_is_refused(self) -> None:
        with acquire_lock_file(self.path, "consumer worker"):
            with self.assertRaises(ServiceBusyError):
                acquire_lock_file(self.path, "consumer worker")

    def test_the_lock_is_reusable_after_release(self) -> None:
        first = acquire_lock_file(self.path, "consumer worker")
        first.release()

        with acquire_lock_file(self.path, "consumer worker"):
            pass

    def test_releasing_twice_is_harmless(self) -> None:
        lease = acquire_lock_file(self.path, "consumer worker")
        lease.release()
        lease.release()

    def test_the_busy_message_names_the_holder(self) -> None:
        with acquire_lock_file(self.path, "consumer worker"):
            try:
                acquire_lock_file(self.path, "consumer worker")
            except ServiceBusyError as exc:
                self.assertIn("consumer worker", str(exc))
            else:
                self.fail("a second lock should not have been granted")

    def test_the_worker_lock_is_separate_from_the_admin_job_lock(self) -> None:
        admin = resolve_path(self.config, self.config.service.job_lock_path)

        self.assertNotEqual(Path(admin), Path(self.path))
        with acquire_lock_file(self.path, "consumer worker"):
            # An admin pipeline run must not be blocked by a healthy worker.
            with acquire_lock_file(admin, "admin job"):
                pass


if __name__ == "__main__":
    unittest.main()
