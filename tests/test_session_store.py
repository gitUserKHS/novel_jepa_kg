from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.session.store import create_session, delete_session, list_sessions, load_session
from src.utils.config import AppConfig


class SessionStoreTests(unittest.TestCase):
    def test_listing_has_no_index_write_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            sessions_path = Path(temporary) / config.chat.session_dir
            sessions_path.mkdir(parents=True)
            index_path = sessions_path / "index.json"
            index_path.write_text("sentinel", encoding="utf-8")

            self.assertEqual(list_sessions(config), [])
            self.assertEqual(index_path.read_text(encoding="utf-8"), "sentinel")

    def test_create_load_list_and_delete_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            created = create_session(config, "테스트 세션", genre="미스터리")

            loaded = load_session(config, created["session_id"])
            self.assertEqual(loaded["title"], "테스트 세션")
            self.assertEqual(list_sessions(config)[0]["session_id"], created["session_id"])

            delete_session(config, created["session_id"])
            self.assertEqual(list_sessions(config), [])


if __name__ == "__main__":
    unittest.main()
