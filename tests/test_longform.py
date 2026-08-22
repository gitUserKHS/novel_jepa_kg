from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.generation.longform import (
    _normalize_section,
    _trim_to_last_sentence,
    assess_section,
    generate_longform,
)
from src.memory.story_rag import load_story_memories
from src.utils.config import AppConfig


OUTLINE_JSON = json.dumps(
    {
        "premise": "아버지가 숨긴 좌표의 진실",
        "ending_intent": "진실을 택하고 등대를 지킨다",
        "beats": [
            {"beat_id": i, "phase": f"{(i - 1) // 2 + 1}막", "purpose": f"비트 {i}", "required_change": f"변화 {i}"}
            for i in range(1, 7)
        ],
    },
    ensure_ascii=False,
)
MEMORY_JSON = json.dumps(
    {
        "title": "폭풍의 밤",
        "summary": "하린이 좌표를 발견한다.",
        "characters": ["서하린", "콩떡"],
        "facts": ["일지에 좌표가 있다"],
        "open_clues": ["좌표의 의미"],
        "resolved_clues": [],
        "locations": ["등대실"],
        "state_changes": ["하린이 결심한다"],
        "state_updates": [{"entity": "서하린", "attribute": "goal", "value": "좌표를 찾는다"}],
        "relations": [{"source": "서하린", "relation": "possesses", "target": "항해일지"}],
        "keywords": ["좌표", "일지"],
    },
    ensure_ascii=False,
)
PROSE_SENTENCES = [
    "서하린은 등대실 창가에 앉아 낡은 항해일지를 펼쳤다.",
    "바람이 유리창을 두드렸고 콩떡은 발치에서 몸을 말았다.",
    "아버지의 필체는 비에 번진 듯 흐릿했지만 숫자만은 또렷했다.",
    "그녀는 좌표를 손끝으로 따라가며 숨을 골랐다.",
    "밖에서는 파도가 검은 바위를 때리는 소리가 이어졌다.",
    "하린은 일지를 덮고 등대의 불빛을 바라보았다.",
]


class FakeClient:
    """프롬프트 종류에 따라 outline / memory / prose 를 돌려주는 가짜 LLM."""

    dry_run = False

    def __init__(self, *, prose_chars: int = 1900, duplicate_prose: bool = False) -> None:
        self.calls = 0
        self.prose_calls = 0
        self.prose_chars = prose_chars
        self.duplicate_prose = duplicate_prose
        self.prompts: list[str] = []

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.7, max_tokens: int = 1200,
             json_mode: bool = False, stream_callback: Any = None, **options: Any) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "계층형 이야기 지도" in prompt:
            return OUTLINE_JSON
        if "확정된 사실만 기록" in prompt:
            return MEMORY_JSON
        self.prose_calls += 1
        seed = 1 if self.duplicate_prose else self.prose_calls
        sentences = []
        index = 0
        while sum(len(s) + 1 for s in sentences) < self.prose_chars:
            base = PROSE_SENTENCES[index % len(PROSE_SENTENCES)]
            sentences.append(f"{base} ({seed}-{index})" if not self.duplicate_prose else base)
            index += 1
        body = "\n\n".join(" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3))
        text = f"### 장면 {seed}-{self.prose_calls}\n\n{body}"
        if stream_callback is not None:
            for start in range(0, len(text), 40):
                stream_callback(text[start : start + 40])
        return text


class StreamRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.chunks: list[str] = []

    def __call__(self, chunk: str) -> None:
        self.chunks.append(chunk)

    def begin_section(self, separator: str = "") -> None:
        self.events.append("begin")

    def restart_section(self, reason: str = "") -> None:
        self.events.append("restart")

    def commit_section(self) -> None:
        self.events.append("commit")

    def abort_section(self) -> None:
        self.events.append("abort")


def _config(root: str) -> AppConfig:
    config = AppConfig(output_root=root)
    config.generation.target_novel_chars = 6000
    config.generation.section_min_chars = 1800
    config.generation.turn_target_chars = 3000
    config.generation.outline_beat_count = 6
    story = Path(root) / "story"
    config.generation.longform_checkpoint_path = str(story / "draft.md")
    config.generation.longform_state_path = str(story / "state.json")
    config.generation.story_memory_path = str(story / "memory.jsonl")
    config.generation.story_ledger_path = str(story / "ledger.json")
    config.generation.story_outline_path = str(story / "outline.json")
    return config


class LongformGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_first_turn_writes_sections_memory_outline_and_state(self) -> None:
        client = FakeClient()
        recorder = StreamRecorder()
        result = generate_longform(
            self.config, client, "세계관", "서하린: 등대지기\n콩떡: 반려견", "첫 장면", stream_callback=recorder,
            return_details=True, continue_existing=False, turn_target_chars=3000, continuation_instruction="시작",
        )
        self.assertIsInstance(result, dict)
        planner = result["planner"]
        self.assertEqual(planner["turn_sections"], 2)
        self.assertFalse(planner["novel_completed"])
        self.assertEqual(planner["story_outline_beats"], 6)
        self.assertEqual(recorder.events.count("commit"), 2)
        self.assertTrue(recorder.chunks)
        draft = Path(self.config.generation.longform_checkpoint_path).read_text(encoding="utf-8")
        self.assertEqual(draft.count("### "), 2)
        memories = load_story_memories(Path(self.config.generation.story_memory_path))
        self.assertEqual([m.section_index for m in memories], [1, 2])
        self.assertEqual(memories[0].summary, "하린이 좌표를 발견한다.")
        state = json.loads(Path(self.config.generation.longform_state_path).read_text(encoding="utf-8"))
        self.assertEqual(state["section_count"], 2)
        self.assertEqual(state["turns_completed"], 1)
        self.assertTrue(Path(self.config.generation.story_outline_path).exists())
        self.assertTrue(Path(self.config.generation.story_ledger_path).exists())
        # 프롬프트는 직전 장 전문이 아니라 꼬리만 싣는다
        second_prompt = [p for p in client.prompts if "이번 장(2장)" in p][0]
        self.assertIn("직전 장면의 끝부분", second_prompt)
        self.assertNotIn("### 장면 1-1", second_prompt)

    def test_continuation_reaches_the_ending(self) -> None:
        client = FakeClient()
        generate_longform(self.config, client, "세계관", "서하린: 등대지기", "첫 장면", return_details=True,
                          continue_existing=False, turn_target_chars=3000)
        result = generate_longform(self.config, client, "세계관", "서하린: 등대지기", "첫 장면", return_details=True,
                                   continue_existing=True, turn_target_chars=3000, continuation_instruction="결말로")
        self.assertTrue(result["planner"]["novel_completed"])
        final_prompt = client.prompts[-2] if "확정된 사실만 기록" in client.prompts[-1] else client.prompts[-1]
        self.assertIn("마지막 장", final_prompt)
        with self.assertRaisesRegex(RuntimeError, "이미 결말"):
            generate_longform(self.config, client, "세계관", "서하린: 등대지기", "첫 장면", return_details=True,
                              continue_existing=True, turn_target_chars=3000)

    def test_duplicate_section_triggers_one_retry(self) -> None:
        client = FakeClient(duplicate_prose=True)
        recorder = StreamRecorder()
        result = generate_longform(self.config, client, "세계관", "서하린: 등대지기", "첫 장면", stream_callback=recorder,
                                   return_details=True, continue_existing=False, turn_target_chars=3000)
        planner = result["planner"]
        self.assertGreaterEqual(planner["turn_stability_retries"], 1)
        self.assertIn("restart", recorder.events)
        self.assertTrue(any("되풀이" in reason or "반복" in reason for reason in planner["retry_reasons"]))


class SectionGateTests(unittest.TestCase):
    def test_normalize_keeps_one_heading_and_body(self) -> None:
        raw = "제목: 버릴 줄\n## 첫 장면\n본문 한 문장.\n\n### 두 번째 소제목\n더 본문."
        section = _normalize_section(raw, 3)
        self.assertTrue(section.startswith("### 첫 장면\n\n"))
        self.assertNotIn("두 번째 소제목", section)
        self.assertIn("더 본문.", section)

    def test_trim_to_last_sentence(self) -> None:
        body = "첫 문장이다. 둘째 문장이다. 셋째 문장이 잘렸"
        self.assertEqual(_trim_to_last_sentence(body), "첫 문장이다. 둘째 문장이다.")

    def test_gate_flags_copy_short_and_repetition(self) -> None:
        body = " ".join(f"{s} {i}" for i, s in enumerate(PROSE_SENTENCES * 8))
        section = f"### 장면\n\n{body}."
        check = assess_section(section, previous_body=body, characters="서하린: 등대지기", prior_titles=[],
                               minimum_chars=200, consumed_beats=[], memory=None, genre="")
        self.assertTrue(any("되풀이" in issue for issue in check.hard))
        short = assess_section("### 장면\n\n짧다.", previous_body="", characters="", prior_titles=[],
                               minimum_chars=500, consumed_beats=[], memory=None, genre="")
        self.assertTrue(any("짧음" in issue for issue in short.hard))
        looped = "### 장면\n\n" + ("등대의 불빛이 바다를 쓸었다. " * 40)
        loop_check = assess_section(looped, previous_body="", characters="", prior_titles=[], minimum_chars=200,
                                    consumed_beats=[], memory=None, genre="")
        self.assertTrue(any("반복" in issue for issue in loop_check.hard))


if __name__ == "__main__":
    unittest.main()
