from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.memory.story_rag import (
    MEMORY_END,
    MEMORY_START,
    StoryMemoryStreamFilter,
    split_story_memory,
    strip_machine_block,
)
from src.service.story_workspace import (
    LiveProseWriter,
    StoryWorkspace,
    clear_live_prose,
    read_live_prose,
)
from src.utils.config import AppConfig


PROSE = "### 차가운 결심의 무게\n\n에블린은 천천히 와인잔을 내려놓았다. 맑고 청아한 울림이 연회장에 퍼졌다."

RECORD = (
    '{\n  "section_index": 1,\n  "title": "차갑게 내려놓은 술잔",\n'
    '  "summary": "에블린이 파혼을 선언한다.",\n  "characters": ["에블린", "칼릭스"],\n'
    '  "facts": ["아르젠이 금지된 예언을 언급했다."],\n  "open_clues": ["예언의 근원"],\n'
    '  "resolved_clues": [],\n  "keywords": ["파혼", "예언"]\n}'
)


def stream(response: str, chunk_size: int = 7) -> str:
    """Feed a response through the filter in chunks, as Ollama would."""
    seen: list[str] = []
    filt = StoryMemoryStreamFilter(seen.append)
    for start in range(0, len(response), chunk_size):
        filt.feed(response[start : start + chunk_size])
    filt.finish()
    return "".join(seen)


class MarkerToleranceTests(unittest.TestCase):
    def test_the_canonical_markers_still_work(self) -> None:
        response = f"{PROSE}\n\n{MEMORY_START}\n{RECORD}\n{MEMORY_END}"

        prose, memory = split_story_memory(response, 1)

        self.assertNotIn("section_index", prose)
        self.assertIn("에블린은 천천히", prose)
        self.assertEqual(memory.summary, "에블린이 파혼을 선언한다.")

    def test_single_angle_bracket_markers_are_recognised(self) -> None:
        """The exact failure seen in the demo: <STORY_MEMORY> ... </STORY_MEMORY>."""
        response = f"{PROSE}\n\n<STORY_MEMORY>\n{RECORD}\n</STORY_MEMORY>"

        prose, memory = split_story_memory(response, 1)

        self.assertNotIn("section_index", prose)
        self.assertNotIn("STORY_MEMORY", prose)
        self.assertEqual(memory.characters, ["에블린", "칼릭스"])

    def test_marker_variants_all_split_cleanly(self) -> None:
        for start_tag, end_tag in [
            ("<STORY_MEMORY>", "</STORY_MEMORY>"),
            ("<<STORY_MEMORY>>", "<<END_STORY_MEMORY>>"),
            ("<<<STORY_MEMORY>>>", "<<<END_STORY_MEMORY>>>"),
            ("< STORY_MEMORY >", "< /STORY_MEMORY >"),
            ("<story_memory>", "</story_memory>"),
        ]:
            with self.subTest(start=start_tag):
                prose, memory = split_story_memory(f"{PROSE}\n{start_tag}{RECORD}{end_tag}", 1)
                self.assertNotIn("section_index", prose)
                self.assertEqual(memory.summary, "에블린이 파혼을 선언한다.")

    def test_a_record_with_no_marker_at_all_is_still_removed(self) -> None:
        prose, memory = split_story_memory(f"{PROSE}\n\n{RECORD}", 1)

        self.assertNotIn("section_index", prose)
        self.assertIn("에블린은 천천히", prose)
        self.assertEqual(memory.summary, "에블린이 파혼을 선언한다.")

    def test_a_truncated_record_does_not_leak(self) -> None:
        """max_tokens can cut the record mid-object, leaving no closing marker."""
        truncated = RECORD[: RECORD.index('"resolved_clues"') + 20]
        response = f"{PROSE}\n\n<STORY_MEMORY>\n{truncated}"

        prose, _memory = split_story_memory(response, 1)

        self.assertNotIn("section_index", prose)
        self.assertNotIn("resolved_clues", prose)
        self.assertIn("에블린은 천천히", prose)

    def test_prose_without_any_record_is_untouched(self) -> None:
        prose, _memory = split_story_memory(PROSE, 1)

        self.assertEqual(prose, PROSE)

    def test_strip_machine_block_keeps_ordinary_braces(self) -> None:
        text = "그녀는 수식 {x}를 떠올렸다. 그것은 아무 의미도 없었다."

        self.assertEqual(strip_machine_block(text), text)


class StreamFilterTests(unittest.TestCase):
    def test_canonical_markers_are_hidden_from_the_stream(self) -> None:
        visible = stream(f"{PROSE}\n{MEMORY_START}\n{RECORD}\n{MEMORY_END}")

        self.assertIn("에블린은 천천히", visible)
        self.assertNotIn("section_index", visible)

    def test_single_bracket_markers_are_hidden_from_the_stream(self) -> None:
        visible = stream(f"{PROSE}\n<STORY_MEMORY>\n{RECORD}\n</STORY_MEMORY>")

        self.assertIn("에블린은 천천히", visible)
        self.assertNotIn("section_index", visible)
        self.assertNotIn("STORY_MEMORY", visible)

    def test_an_unmarked_record_is_hidden_from_the_stream(self) -> None:
        visible = stream(f"{PROSE}\n\n{RECORD}")

        self.assertIn("에블린은 천천히", visible)
        self.assertNotIn("section_index", visible)

    def test_a_truncated_record_is_hidden_from_the_stream(self) -> None:
        truncated = RECORD[: RECORD.index('"open_clues"') + 15]
        visible = stream(f"{PROSE}\n<STORY_MEMORY>\n{truncated}")

        self.assertIn("에블린은 천천히", visible)
        self.assertNotIn("open_clues", visible)

    def test_plain_prose_streams_through_completely(self) -> None:
        self.assertEqual(stream(PROSE), PROSE)

    def test_a_brace_in_prose_is_not_swallowed(self) -> None:
        text = PROSE + " 그는 {괄호}를 보았다. " + "이야기는 계속되었다. " * 40

        self.assertEqual(stream(text), text)


class LiveProseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        # Story ids are validated against a 32-char lowercase hex pattern.
        self.workspace = StoryWorkspace.for_story(self.config, "a1b2c3d4" * 4, create=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_nothing_is_reported_before_generation_starts(self) -> None:
        self.assertEqual(read_live_prose(self.workspace), "")

    def test_streamed_chunks_become_readable_prose(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=8)

        for chunk in ["에블린은 ", "천천히 ", "와인잔을 ", "내려놓았다."]:
            writer.feed(chunk)
        writer.flush()

        self.assertEqual(read_live_prose(self.workspace), "에블린은 천천히 와인잔을 내려놓았다.")

    def test_small_writes_are_batched_until_flush(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1000)

        writer.feed("짧은 조각")

        self.assertEqual(read_live_prose(self.workspace), "")
        writer.flush()
        self.assertEqual(read_live_prose(self.workspace), "짧은 조각")

    def test_reset_clears_the_committed_tail(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)
        writer.feed("이전 턴의 잔여물")

        writer.reset()

        self.assertEqual(read_live_prose(self.workspace), "")
        self.assertFalse(self.workspace.live.exists())

    def test_clearing_an_absent_file_is_not_an_error(self) -> None:
        clear_live_prose(self.workspace)

        self.assertEqual(read_live_prose(self.workspace), "")

    def test_it_satisfies_the_generator_section_stream_protocol(self) -> None:
        """Without these the generator defers streaming and live.txt stays empty."""
        from src.generation.hallucination import _has_section_stream_control

        writer = LiveProseWriter(self.workspace)

        self.assertTrue(_has_section_stream_control(writer))

    def test_a_committed_section_clears_the_live_file(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)
        writer.begin_section("\n\n")
        writer.feed("완성된 섹션 본문")

        writer.commit_section()

        self.assertEqual(read_live_prose(self.workspace), "")

    def test_a_restarted_section_replaces_rather_than_appends(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)
        writer.begin_section()
        writer.feed("버려질 초안")

        writer.restart_section("반복 때문에 다시 씀")
        writer.feed("새 초안")
        writer.flush()

        self.assertEqual(read_live_prose(self.workspace), "새 초안")

    def test_an_aborted_section_leaves_nothing_behind(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)
        writer.begin_section()
        writer.feed("실패한 섹션")

        writer.abort_section()

        self.assertEqual(read_live_prose(self.workspace), "")

    def test_calling_the_writer_directly_streams(self) -> None:
        writer = LiveProseWriter(self.workspace, flush_chars=1)

        writer("직접 호출된 조각")

        self.assertEqual(read_live_prose(self.workspace), "직접 호출된 조각")

    def test_live_path_stays_inside_the_story_directory(self) -> None:
        self.assertEqual(self.workspace.live.parent, self.workspace.root)
        self.assertEqual(Path(self.workspace.live).name, "live.txt")


if __name__ == "__main__":
    unittest.main()
