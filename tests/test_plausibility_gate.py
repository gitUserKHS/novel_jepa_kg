"""개연성 장치의 계약.

- 장을 쓰기 전에 설계(목표·장애·전환·결과)를 JSON 으로 받아 작가 프롬프트에 넣는다.
- 쓴 뒤 연속성 편집자 역할의 모델이 확정 사실·앞선 사건과의 모순을 검토한다.
- 모순이 있거나 점수가 기준 미만이면 문제 목록을 들고 한 번 고쳐 쓰고 다시 검토한다. 고친 판이 더 낫지 않으면
  처음 판을 채택하고 그 결정을 기록한다.
- 검토·설계 호출이 실패하면 본문을 막지 않는다.
- 프롬프트는 문맥 창 예산(llm.num_ctx) 안에 들어간다.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.generation.longform import (
    PLAN_ROLE_MARKER,
    REPAIR_MARKER,
    generate_longform,
    prompt_char_budget,
    section_max_tokens,
)
from src.generation.plausibility import DIALOGUE_ROLE_MARKER, REVIEW_ROLE_MARKER, PlausibilityReport, parse_report
from src.memory.story_rag import load_story_memories
from src.service.runtime import make_llm_client
from src.utils.config import AppConfig

MEMORY_JSON = json.dumps({"title": "t", "summary": "요약.", "characters": ["서하린"], "facts": [], "keywords": ["k"]},
                         ensure_ascii=False)
PLAN_JSON = json.dumps({
    "goal": "하린이 좌표의 출처를 확인한다", "obstacle": "일지의 마지막 장이 찢겨 있다",
    "turn": "콩떡이 찢긴 종이를 물어 온다", "outcome": "하린이 좌표가 아버지의 마지막 항로임을 안다",
    "uses": ["항해일지의 좌표"], "avoid": ["좌표를 처음 발견하는 장면"],
}, ensure_ascii=False)
GOOD_REVIEW = json.dumps({"score": 9, "contradictions": [], "unmotivated": [], "continuity_ok": True,
                          "beat_progress": True, "verdict": "모순 없음"}, ensure_ascii=False)
BAD_REVIEW = json.dumps({"score": 4, "contradictions": ["콩떡이 3장에서 등대에 있는데 여기선 배 위에 있다"],
                         "unmotivated": ["하린이 이유 없이 좌표를 태운다"], "continuity_ok": True,
                         "beat_progress": True, "verdict": "장소가 어긋난다"}, ensure_ascii=False)
WORSE_REVIEW = json.dumps({"score": 3, "contradictions": ["콩떡의 위치", "일지의 소유자"], "unmotivated": [],
                           "continuity_ok": False, "beat_progress": False, "verdict": "더 나빠졌다"}, ensure_ascii=False)


def _prose(marker: str, chars: int = 1900) -> str:
    """겹치는 12자 구절이 없는 산문 — 게이트의 반복 검사를 통과해야 하므로 문장마다 음절을 달리 만든다."""
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


class RoleClient:
    """역할 표식으로 호출을 구분한다: 설계 → PLAN_JSON, 검토 → 예약된 답, 기록 → MEMORY_JSON, 그 외 → 예약된 본문."""

    dry_run = False

    def __init__(self, drafts: list[str], reviews: list[str] | None = None, *, plan: str = PLAN_JSON,
                 dialogue: str = '{"non_speakers": [], "dialogue": []}') -> None:
        self.drafts = list(drafts)
        self.reviews = list(reviews or [])
        self.plan = plan
        self.dialogue = dialogue
        self.prose_prompts: list[str] = []
        self.plan_prompts: list[str] = []
        self.review_prompts: list[str] = []
        self.dialogue_prompts: list[str] = []
        self.memory_prompts: list[str] = []
        self.systems: set[str] = set()
        self.extra = 0

    def chat(self, prompt: str, system: str | None = None, stream_callback: Any = None, **_options: Any) -> str:
        self.systems.add(str(system))
        if PLAN_ROLE_MARKER in prompt:
            self.plan_prompts.append(prompt)
            return self.plan
        if DIALOGUE_ROLE_MARKER in prompt:
            self.dialogue_prompts.append(prompt)
            return self.dialogue
        if REVIEW_ROLE_MARKER in prompt:
            self.review_prompts.append(prompt)
            return self.reviews.pop(0) if self.reviews else GOOD_REVIEW
        if "[역할: 기록 담당]" in prompt:
            self.memory_prompts.append(prompt)
            return MEMORY_JSON
        self.prose_prompts.append(prompt)
        if self.drafts:
            text = self.drafts.pop(0)
        else:
            self.extra += 1
            text = _prose(f"extra{self.extra}")
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

    def revise_section(self, reason: str = "") -> None:
        self.events.append(("revise", reason))

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
    config.generation.turn_target_chars = 1000
    story = Path(root) / "story"
    config.generation.longform_checkpoint_path = str(story / "draft.md")
    config.generation.longform_state_path = str(story / "state.json")
    config.generation.story_memory_path = str(story / "memory.jsonl")
    config.generation.story_ledger_path = str(story / "ledger.json")
    config.generation.story_outline_path = str(story / "outline.json")
    return config


class ReportRuleTests(unittest.TestCase):
    def test_parse_tolerates_fences_strings_floats_and_truncation(self) -> None:
        report = parse_report('```json\n{"score": "7.6", "contradictions": ["a"], "continuity_ok": "false", "verdict": " x  y "}\n```')
        assert report is not None
        self.assertEqual((report.score, report.contradictions, report.continuity_ok, report.verdict), (8, ["a"], False, "x y"))
        truncated = parse_report('{"score": 6, "contradictions": ["장소 불일치"], "unmotivated": ["갑작스런 등장", "미완성 항')
        assert truncated is not None
        self.assertEqual(truncated.score, 6)
        self.assertEqual(truncated.contradictions, ["장소 불일치"])
        self.assertIsNone(parse_report("검토 결과: 문제 없음"))
        self.assertIsNone(parse_report('{"verdict": "점수가 없다"}'))

    def test_a_line_spoken_by_something_that_cannot_speak_becomes_a_contradiction(self) -> None:
        raw = json.dumps({
            "score": 9, "contradictions": [], "unmotivated": [],
            "dialogue": [
                {"quote": "제발, 조금만 더", "speaker": "서하린", "can_speak": True},
                {"quote": "하린아, 으르렁!", "speaker": "콩떡", "can_speak": False},
                {"quote": "여긴... 안 되는", "speaker": "서하린", "can_speak": "true"},
            ],
        }, ensure_ascii=False)
        report = parse_report(raw)
        assert report is not None
        self.assertEqual(report.contradictions, ["대사 '하린아, 으르렁!' 가 말할 수 없는 존재(콩떡)의 말로 읽힘"])
        self.assertFalse(report.passes(7), "an impossible speaker fails the gate even at a high score")
        clean = parse_report('{"score": 9, "dialogue": [{"quote": "응", "speaker": "서하린", "can_speak": true}]}')
        assert clean is not None
        self.assertEqual(clean.contradictions, [])

    def test_contradictions_always_fail_and_low_scores_expose_the_rest(self) -> None:
        clean = PlausibilityReport(score=8, unmotivated=["작은 우연"], beat_progress=False)
        self.assertTrue(clean.passes(7))
        self.assertEqual(len(clean.notes()), 2, "soft observations are kept for the record")
        contradiction = PlausibilityReport(score=9, contradictions=["죽은 인물이 말한다"])
        self.assertFalse(contradiction.passes(7))
        self.assertIn("모순", contradiction.issues(7)[0])
        low = PlausibilityReport(score=5, unmotivated=["갑자기 폭발"], continuity_ok=False, beat_progress=False)
        issues = low.issues(7)
        self.assertEqual(len(issues), 3)
        bare = PlausibilityReport(score=5, verdict="흐릿하다")
        self.assertIn("5/10", bare.issues(7)[0])
        self.assertTrue(PlausibilityReport.unavailable("down").passes(7), "an unavailable review never blocks")


class PlausibilityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, client: Any, recorder: Recorder | None = None, *, turn_chars: int = 1000,
             previous_scene: str = "첫 장면") -> dict[str, Any]:
        result = generate_longform(
            self.config, client, "세계관", "서하린: 등대지기\n콩떡: 반려견", previous_scene,
            stream_callback=recorder or Recorder(), return_details=True, continue_existing=False,
            turn_target_chars=turn_chars, continuation_instruction="좌표의 출처를 밝힌다",
        )
        assert isinstance(result, dict)
        return result

    def _draft(self) -> str:
        return Path(self.config.generation.longform_checkpoint_path).read_text(encoding="utf-8")

    def test_plan_is_asked_first_and_shapes_the_writing_and_review_prompts(self) -> None:
        client = RoleClient([_prose("first")])
        result = self._run(client)
        self.assertEqual(len(client.plan_prompts), 1)
        self.assertEqual(len(client.prose_prompts), 1)
        write_prompt = client.prose_prompts[0]
        self.assertIn("[이번 장의 설계 — 이 인과를 따른다]", write_prompt)
        self.assertIn("목표: 하린이 좌표의 출처를 확인한다", write_prompt)
        self.assertIn("되풀이 금지: 좌표를 처음 발견하는 장면", write_prompt)
        self.assertIn("좌표의 출처를 밝힌다", write_prompt, "the user's direction stays in the task")
        self.assertIn("하린이 좌표가 아버지의 마지막 항로임을 안다", client.review_prompts[0], "the review checks the plan")
        self.assertEqual(len(client.systems), 1, "one shared system prompt keeps the prefix cacheable")
        plan_prompt, write_prompt = client.plan_prompts[0], client.prose_prompts[0]
        shared = plan_prompt[: plan_prompt.index("[이번 장(")]
        self.assertTrue(write_prompt.startswith(shared), "plan, write and review share the canon prefix")
        self.assertTrue(client.review_prompts[0].startswith(shared))
        self.assertEqual(result["planner"]["plausibility_by_section"], {1: 9})
        self.assertEqual(result["planner"]["mean_plausibility"], 9.0)
        self.assertTrue(any("개연성 9/10" in line for line in result["planner"]["turn_retry_reasons"]))

    def test_speaker_check_runs_only_when_there_is_dialogue_and_feeds_the_gate(self) -> None:
        # _prose() 에는 따옴표가 없다 → 화자 점검 호출 없음.
        silent = RoleClient([_prose("quiet")])
        self._run(silent)
        self.assertEqual(silent.dialogue_prompts, [])
        # 대사가 있으면 점검 호출이 한 번 나가고, 말할 수 없는 화자는 검토 점수와 무관하게 고쳐 쓰기를 부른다.
        talking = _prose("dog") + '\n\n"하린아, 으르렁!"\n\n콩떡이 문 쪽을 향해 짖었다.'
        client = RoleClient(
            [talking, _prose("fixed")], [GOOD_REVIEW, GOOD_REVIEW],
            dialogue=json.dumps({"non_speakers": ["콩떡"], "dialogue": [
                {"quote": "하린아, 으르렁!", "speaker": "콩떡", "can_speak": False}]}, ensure_ascii=False),
        )
        # 고친 판에는 대사가 없으니 두 번째 점검은 나가지 않는다.
        result = self._run(client)
        self.assertEqual(len(client.dialogue_prompts), 1)
        self.assertIn(DIALOGUE_ROLE_MARKER, client.dialogue_prompts[0])
        self.assertIn("으르렁", client.dialogue_prompts[0])
        self.assertEqual(result["planner"]["turn_plausibility_repairs"], 1)
        self.assertEqual(result["planner"]["turn_plausibility_repair_successes"], 1)
        self.assertIn("fixed", self._draft())
        self.assertTrue(any("말할 수 없는 존재(콩떡)" in line for line in result["planner"]["turn_retry_reasons"]))
        self.assertIn("말할 수 없는 존재(콩떡)", client.prose_prompts[1], "the repair prompt names the offending line")

    def test_a_broken_plan_does_not_stop_the_section(self) -> None:
        client = RoleClient([_prose("first")], plan="설계를 못 하겠다")
        self._run(client)
        self.assertNotIn("[이번 장의 설계", client.prose_prompts[0])
        self.assertIn("first", self._draft())

    def test_contradiction_triggers_one_repair_that_is_reviewed_again(self) -> None:
        client = RoleClient([_prose("first"), _prose("repaired")], [BAD_REVIEW, GOOD_REVIEW])
        recorder = Recorder()
        result = self._run(client, recorder)
        planner = result["planner"]
        self.assertEqual(len(client.prose_prompts), 2, "write once, repair once")
        self.assertEqual(len(client.review_prompts), 2, "the repaired draft is reviewed again")
        repair_prompt = client.prose_prompts[1]
        self.assertIn(REPAIR_MARKER, repair_prompt)
        self.assertIn("콩떡이 3장에서 등대에 있는데", repair_prompt, "the exact contradiction is handed to the writer")
        self.assertIn("first", repair_prompt, "repair revises the draft in place instead of starting over")
        self.assertIn("repaired", self._draft())
        self.assertNotIn("장면 first", self._draft())
        self.assertEqual(planner["turn_plausibility_repairs"], 1)
        self.assertEqual(planner["turn_plausibility_repair_successes"], 1)
        self.assertEqual(planner["plausibility_by_section"], {1: 9})
        self.assertTrue(any("모순" in line for line in planner["turn_retry_reasons"]))
        self.assertIn("repaired", client.memory_prompts[0], "memory comes from the repaired draft")
        revisions = [event for event in recorder.events if event[0] == "revise"]
        self.assertEqual(len(revisions), 1)
        self.assertIn("모순", revisions[0][1])
        decisions = [event for event in recorder.events if event[0] == "note" and event[1] == "decision"]
        self.assertIn("고친 판을 채택", decisions[0][2])
        self.assertIn("first", decisions[0][3], "the pre-repair draft travels to the display only")
        memories = load_story_memories(Path(self.config.generation.story_memory_path))
        self.assertEqual([m.section_index for m in memories], [1])

    def test_a_worse_repair_keeps_the_original_and_says_so(self) -> None:
        client = RoleClient([_prose("first"), _prose("repaired")], [BAD_REVIEW, WORSE_REVIEW])
        recorder = Recorder()
        result = self._run(client, recorder)
        self.assertIn("first", self._draft())
        self.assertNotIn("repaired", self._draft())
        self.assertEqual(result["planner"]["turn_plausibility_repairs"], 1)
        self.assertEqual(result["planner"]["turn_plausibility_repair_successes"], 0)
        self.assertEqual(result["planner"]["plausibility_by_section"], {1: 4}, "the kept draft keeps its own score")
        decisions = [event for event in recorder.events if event[0] == "note" and event[1] == "decision"]
        self.assertIn("처음 판을 그대로 채택", decisions[0][2])
        self.assertIn("repaired", decisions[0][3])

    def test_an_unreadable_review_does_not_block_the_section(self) -> None:
        client = RoleClient([_prose("first")], ["검토 불가"])
        result = self._run(client)
        self.assertEqual(len(client.prose_prompts), 1)
        self.assertIn("first", self._draft())
        self.assertEqual(result["planner"]["plausibility_by_section"], {})
        self.assertEqual(result["planner"]["mean_plausibility"], 0.0)
        self.assertTrue(any("검토 불가" in line for line in result["planner"]["turn_retry_reasons"]))

    def test_a_repair_that_breaks_the_mechanical_gate_is_rejected(self) -> None:
        client = RoleClient([_prose("first"), "### 장면\n\n너무 짧다."], [BAD_REVIEW, GOOD_REVIEW])
        result = self._run(client)
        self.assertIn("first", self._draft())
        self.assertEqual(result["planner"]["turn_plausibility_repair_successes"], 0)

    def test_the_gate_and_the_plan_can_be_switched_off(self) -> None:
        self.config.generation.enable_plausibility_gate = False
        self.config.generation.enable_scene_plan = False
        client = RoleClient([_prose("first")])
        result = self._run(client)
        self.assertEqual((client.plan_prompts, client.review_prompts), ([], []))
        self.assertEqual(result["planner"]["turn_plausibility_repairs"], 0)
        self.assertNotIn("[이번 장의 설계", client.prose_prompts[0])

    def test_prompts_stay_inside_the_context_budget(self) -> None:
        self.config.llm.num_ctx = 4000
        self.config.generation.enable_story_memory_rag = True
        client = RoleClient([_prose("first")])
        self._run(client, previous_scene=_prose("previous", chars=6000))
        budget = prompt_char_budget(self.config, section_max_tokens(self.config, 1800))
        self.assertLess(budget, 6000, "the tiny window forces trimming")
        for prompt in (*client.plan_prompts, *client.prose_prompts):
            self.assertLessEqual(len(prompt), budget, "the direction and the previous tail are trimmed to fit")
        self.assertIn("직전 장면의 끝부분", client.prose_prompts[0])
        self.assertIn("하린은 등대를 바라보았다", client.prose_prompts[0], "the tail keeps its end, not its start")

    def test_dry_run_client_walks_the_whole_pipeline(self) -> None:
        self.config.llm.dry_run = True
        self.config.generation.enable_story_outline = True
        client = make_llm_client(self.config)
        result = self._run(client, turn_chars=3000)
        planner = result["planner"]
        self.assertEqual(planner["turn_sections"], 2)
        self.assertEqual(planner["plausibility_by_section"], {1: 8, 2: 8})
        self.assertEqual(planner["turn_plausibility_repairs"], 0)
        self.assertEqual(planner["story_outline_beats"], 4, "the dry-run outline is used, not the fallback spine")


if __name__ == "__main__":
    unittest.main()
