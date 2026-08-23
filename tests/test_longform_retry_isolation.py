"""게이트가 장을 버리고 다시 쓸 때의 계약.

- 반복 루프만 문제면 되풀이된 문장을 걷어내고 그대로 채택한다 (재생성 없음, 전개 유지).
- 다시 쓸 때 버려진 초안의 본문은 재생성 프롬프트·메모리 추출·다음 장 문맥·원고 어디에도 들어가지 않는다.
- 다시 쓴 판이 더 낫지 않으면 처음 초안을 채택하고, 그 결정을 스트림에 알린다.
- 스트림 콜백은 사유와 버려진 초안을 받는다 (화면 표시용).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.generation.longform import generate_longform, trim_repetitions
from src.memory.story_rag import load_story_memories
from src.utils.config import AppConfig

MEMORY_JSON = json.dumps({"title": "t", "summary": "요약.", "characters": ["서하린"], "facts": [], "keywords": ["k"]},
                         ensure_ascii=False)
DISCARDED_MARK = "버려진초안표식문장이다"
KEPT_MARK = "채택된초안표식문장이다"
LOOP = "등대의 불빛이 검은 바다를 천천히 쓸었다."


def _prose(marker: str, chars: int = 1900) -> str:
    """겹치는 12자 구절이 없는 산문 (게이트의 반복 검사를 통과해야 하므로 문장마다 음절을 달리 만든다)."""
    seed = sum(ord(ch) for ch in marker) or 1
    out: list[str] = []
    index = 0
    while sum(len(s) + 1 for s in out) < chars:
        syllables = "".join(chr(0xAC00 + ((seed * 31 + index * 131 + j * 7919) % 11172)) for j in range(18))
        tail = f" {marker}" if index == 0 else ""
        out.append(f"{syllables}{tail} 했다.")
        index += 1
    out.append(f"끝으로 {marker} 하린은 등대를 바라보았다.")
    body = "\n\n".join(" ".join(out[i : i + 3]) for i in range(0, len(out), 3))
    return f"### 장면 {marker}\n\n{body}"


class ScriptedClient:
    """prose 호출마다 미리 정한 본문을 돌려준다. 메모리 호출은 고정 JSON."""

    dry_run = False

    def __init__(self, drafts: list[str]) -> None:
        self.drafts = list(drafts)
        self.prompts: list[str] = []
        self.prose_prompts: list[str] = []
        self.memory_prompts: list[str] = []
        self.extra_calls = 0

    def chat(self, prompt: str, system: str | None = None, stream_callback: Any = None, **_options: Any) -> str:
        self.prompts.append(prompt)
        if "확정된 사실만 기록" in prompt:
            self.memory_prompts.append(prompt)
            return MEMORY_JSON
        self.prose_prompts.append(prompt)
        if self.drafts:
            text = self.drafts.pop(0)
        else:
            self.extra_calls += 1
            text = _prose(f"extra{self.extra_calls}")
        if stream_callback is not None:
            for start in range(0, len(text), 40):
                stream_callback(text[start : start + 40])
        return text


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []

    def __call__(self, chunk: str) -> None:
        pass

    def begin_section(self, separator: str = "") -> None:
        self.events.append(("begin",))

    def restart_section(self, reason: str = "") -> None:
        self.events.append(("restart", reason))

    def note_section(self, kind: str, text: str, discarded: str = "") -> None:
        self.events.append(("note", kind, text, discarded))

    def commit_section(self) -> None:
        self.events.append(("commit",))

    def abort_section(self) -> None:
        self.events.append(("abort",))


def _config(root: str) -> AppConfig:
    config = AppConfig(output_root=root)
    config.generation.enable_story_outline = False
    config.generation.enable_story_memory_rag = False
    config.generation.enable_consumed_beat_ledger = False
    config.generation.target_novel_chars = 20000
    config.generation.section_min_chars = 1800
    config.generation.turn_target_chars = 3000
    story = Path(root) / "story"
    config.generation.longform_checkpoint_path = str(story / "draft.md")
    config.generation.longform_state_path = str(story / "state.json")
    config.generation.story_memory_path = str(story / "memory.jsonl")
    config.generation.story_ledger_path = str(story / "ledger.json")
    config.generation.story_outline_path = str(story / "outline.json")
    return config


class TrimRepetitionsTests(unittest.TestCase):
    def test_duplicate_sentences_and_paragraphs_are_kept_once(self) -> None:
        paragraph = "둘째 문단이다. 아무튼 여기는 바람이 세게 분다."
        section = ("### 장면\n\n첫 문장이다. " + (LOOP + " ") * 5 + "\n\n" + paragraph + '\n"응." "응."\n\n'
                   + paragraph + "\n\n" + LOOP + " 마지막 문장이다.")
        cleaned, removed = trim_repetitions(section)
        self.assertEqual(removed, 4 + 1 + 1, "4 looped copies, 1 duplicate paragraph, 1 trailing copy")
        self.assertEqual(cleaned.count(LOOP), 1)
        self.assertEqual(cleaned.count("아무튼 여기는"), 1)
        self.assertIn('"응." "응."', cleaned, "short interjections may repeat")
        self.assertIn("마지막 문장이다.", cleaned)
        self.assertTrue(cleaned.startswith("### 장면\n\n첫 문장이다. " + LOOP))

    def test_clean_text_is_untouched(self) -> None:
        section = _prose("ok")
        cleaned, removed = trim_repetitions(section)
        self.assertEqual(removed, 0)
        self.assertEqual(cleaned, section)


class RetryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, client: ScriptedClient, recorder: Recorder, turn_chars: int = 1000) -> dict[str, Any]:
        self.config.generation.turn_target_chars = turn_chars
        result = generate_longform(
            self.config, client, "세계관", "서하린: 등대지기\n콩떡: 반려견", "첫 장면", stream_callback=recorder,
            return_details=True, continue_existing=False, turn_target_chars=turn_chars, continuation_instruction="시작",
        )
        assert isinstance(result, dict)
        return result

    def _draft(self) -> str:
        return Path(self.config.generation.longform_checkpoint_path).read_text(encoding="utf-8")

    def test_repetition_loop_is_trimmed_instead_of_rewritten(self) -> None:
        looped = _prose("loop", chars=1500) + "\n\n" + " ".join([LOOP] * 9)
        client = ScriptedClient([looped])
        recorder = Recorder()
        result = self._run(client, recorder)
        planner = result["planner"]
        self.assertEqual(len(client.prose_prompts), 1, "no second generation call")
        self.assertEqual(planner["turn_repetition_trims"], 1)
        self.assertEqual(planner["turn_stability_retries"], 0)
        self.assertTrue(any("걷어냄" in reason for reason in planner["turn_retry_reasons"]))
        draft = self._draft()
        self.assertEqual(draft.count(LOOP), 1, "the loop survives exactly once")
        self.assertIn("끝으로 loop 하린은", draft, "the rest of the draft -- the story -- is kept")
        self.assertIn(("note", "trim", "반복된 문장 8개를 걷어내고 이어가.", ""), recorder.events)
        self.assertNotIn("restart", [event[0] for event in recorder.events])
        self.assertEqual(client.memory_prompts[0].count(LOOP), 1, "memory is extracted from the trimmed prose")

    def test_loop_that_hit_the_token_cap_is_trimmed_and_its_tail_closed(self) -> None:
        # 반복 + 마지막 문장 미완: 루프가 토큰 상한까지 돈 전형적인 모양. 걷어낸 뒤 마지막 완결 문장까지 되돌린다.
        looped = _prose("cap", chars=1500) + "\n\n" + " ".join([LOOP] * 9) + " 그리고 문장이 끊겨"
        client = ScriptedClient([looped])
        recorder = Recorder()
        result = self._run(client, recorder)
        self.assertEqual(len(client.prose_prompts), 1, "no rewrite")
        self.assertEqual(result["planner"]["turn_repetition_trims"], 1)
        draft = self._draft()
        self.assertEqual(draft.count(LOOP), 1)
        self.assertNotIn("문장이 끊겨", draft, "the unfinished tail is dropped")
        self.assertIn("끝으로 cap 하린은", draft)

    def test_trim_that_leaves_too_little_falls_back_to_a_rewrite(self) -> None:
        mostly_loop = "### 장면 loop\n\n" + " ".join([LOOP] * 40) + " 하린은 등대를 올려다보았다."
        client = ScriptedClient([mostly_loop, _prose(KEPT_MARK)])
        recorder = Recorder()
        result = self._run(client, recorder)
        self.assertEqual(len(client.prose_prompts), 2)
        self.assertEqual(result["planner"]["turn_repetition_trims"], 0)
        self.assertEqual(result["planner"]["turn_stability_retries"], 1)
        self.assertIn(KEPT_MARK, self._draft())
        self.assertNotIn(LOOP, self._draft())

    def test_discarded_draft_never_reaches_prompts_memory_or_the_next_section(self) -> None:
        short_draft = f"### 장면 short\n\n{DISCARDED_MARK}. 너무 짧은 초안이다."
        client = ScriptedClient([short_draft, _prose(KEPT_MARK), _prose("second")])
        recorder = Recorder()
        result = self._run(client, recorder, turn_chars=3000)
        planner = result["planner"]
        self.assertEqual(planner["turn_stability_retries"], 1)
        self.assertEqual(planner["turn_stability_retry_successes"], 1)
        self.assertEqual(planner["turn_sections"], 2)

        retry_prompt = client.prose_prompts[1]
        self.assertIn("[이전 시도의 문제", retry_prompt)
        self.assertIn("본문이 너무 짧음", retry_prompt)
        self.assertNotIn(DISCARDED_MARK, retry_prompt, "the discarded draft's text is not in the retry prompt")
        self.assertNotIn("너무 짧은 초안이다", retry_prompt)

        self.assertTrue(client.memory_prompts)
        self.assertNotIn(DISCARDED_MARK, client.memory_prompts[0])
        self.assertIn(KEPT_MARK, client.memory_prompts[0], "memory comes from the kept draft")

        next_prompt = client.prose_prompts[2]
        self.assertIn("직전 장면의 끝부분", next_prompt)
        self.assertIn(KEPT_MARK, next_prompt, "the next section continues from the kept draft")
        self.assertNotIn(DISCARDED_MARK, next_prompt)

        self.assertNotIn(DISCARDED_MARK, self._draft())
        memories = load_story_memories(Path(self.config.generation.story_memory_path))
        self.assertEqual([m.section_index for m in memories], [1, 2])

        restarts = [event for event in recorder.events if event[0] == "restart"]
        self.assertEqual(len(restarts), 1)
        self.assertIn("본문이 너무 짧음", restarts[0][1], "the reason is handed to the stream for display")
        decisions = [event for event in recorder.events if event[0] == "note" and event[1] == "decision"]
        self.assertEqual(len(decisions), 1)
        self.assertIn("다시 쓴 판을 채택", decisions[0][2])
        self.assertIn(DISCARDED_MARK, decisions[0][3], "the discarded draft travels to the display only")

    def test_worse_rewrite_keeps_the_first_draft_and_says_so(self) -> None:
        first = f"### 장면 first\n\n{KEPT_MARK}. 짧다."
        second = f"### 장면 second\n\n{DISCARDED_MARK}. 역시 짧다."
        client = ScriptedClient([first, second])
        recorder = Recorder()
        result = self._run(client, recorder)
        self.assertEqual(result["planner"]["turn_stability_retries"], 1)
        self.assertEqual(result["planner"]["turn_stability_retry_successes"], 0)
        self.assertIn(KEPT_MARK, self._draft())
        self.assertNotIn(DISCARDED_MARK, self._draft())
        decisions = [event for event in recorder.events if event[0] == "note" and event[1] == "decision"]
        self.assertIn("처음 초안을 그대로 채택", decisions[0][2])
        self.assertIn(DISCARDED_MARK, decisions[0][3])

    def test_stability_retry_can_be_switched_off(self) -> None:
        self.config.generation.enable_stability_retry = False
        client = ScriptedClient([f"### 장면 short\n\n{KEPT_MARK}. 짧다."])
        result = self._run(client, Recorder())
        self.assertEqual(result["planner"]["turn_stability_retries"], 0)
        self.assertIn(KEPT_MARK, self._draft())


if __name__ == "__main__":
    unittest.main()
