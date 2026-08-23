from __future__ import annotations

import tempfile
import unittest

from src.service.story_workspace import (
    LiveProseWriter,
    StoryWorkspace,
    read_live_note,
    read_live_prose,
    write_live_note,
)
from src.utils.config import AppConfig


class LiveNoteTests(unittest.TestCase):
    """화면이 읽는 live_note.json: 왜 실시간 본문이 사라졌는지 (사유·버려진 초안·걷어낸 문장·채택 결정)."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.workspace = StoryWorkspace.for_story(self.config, "a" * 32, create=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_restart_keeps_the_discarded_draft_beside_the_reason(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1000)
        writer.begin_section()
        writer.feed("### 4장\n\n같은 문장이 계속 ")
        writer.feed("돌았다.")
        writer.restart_section("같은 구절이 6회 반복됨")
        note = read_live_note(self.workspace)
        self.assertEqual(note["kind"], "retry")
        self.assertEqual(note["text"], "같은 구절이 6회 반복됨")
        self.assertEqual(note["discarded"], "### 4장\n\n같은 문장이 계속 돌았다.", "unflushed chunks are included")
        self.assertEqual(read_live_prose(self.workspace), "", "the live prose restarts from scratch")
        writer.feed("### 4장\n\n다시 쓴 본문.")
        writer.flush()
        self.assertEqual(read_live_prose(self.workspace), "### 4장\n\n다시 쓴 본문.")
        self.assertEqual(read_live_note(self.workspace)["kind"], "retry", "the note stays while the retry streams")

    def test_notes_survive_commit_and_clear_on_the_next_section(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)
        writer.begin_section()
        writer.feed("본문")
        writer.note_section("decision", "처음 초안을 그대로 채택했어.", "버려진 판")
        writer.commit_section()
        self.assertEqual(read_live_prose(self.workspace), "")
        note = read_live_note(self.workspace)
        self.assertEqual((note["kind"], note["discarded"]), ("decision", "버려진 판"))
        writer.begin_section()
        self.assertIsNone(read_live_note(self.workspace), "a new section starts clean")
        writer.note_section("trim", "반복된 문장 3개를 걷어내고 이어가.")
        self.assertEqual(read_live_note(self.workspace)["text"], "반복된 문장 3개를 걷어내고 이어가.")
        writer.reset()
        self.assertIsNone(read_live_note(self.workspace), "job start / abort wipes everything")

    def test_abort_and_reset_wipe_the_note(self) -> None:
        writer = LiveProseWriter(self.workspace)
        writer.begin_section()
        writer.feed("x")
        writer.restart_section("이유")
        writer.abort_section()
        self.assertIsNone(read_live_note(self.workspace))
        self.assertEqual(read_live_prose(self.workspace), "")

    def test_garbage_note_reads_as_none(self) -> None:
        self.workspace.note.write_text("not json", encoding="utf-8")
        self.assertIsNone(read_live_note(self.workspace))
        write_live_note(self.workspace, "", "no kind")
        self.assertIsNone(read_live_note(self.workspace))
        self.workspace.reset()
        self.assertFalse(self.workspace.note.exists())


if __name__ == "__main__":
    unittest.main()
