from __future__ import annotations

import json
import tempfile
import unittest

from src.memory.story_rag import StoryMemory, load_story_memories, write_story_memories
from src.service.consumer_store import AuthorizationError, ConsumerStore, StoryBusyError
from src.service.story_editor import (
    QUICK_REWRITES,
    StoryEditError,
    length_bounds,
    list_sections,
    normalize_edit,
    plan_edits,
    refresh_section_memory,
    replace_section,
    retry_note_for,
    rewrite_messages,
    rewrite_plan,
    rewrite_temperature,
    section_memory,
    stream_rewrite,
    update_section_memory,
)
from src.service.story_sheets import character_sheet, style_guide, world_sheet
from src.service.story_workspace import StoryWorkspace
from src.utils.config import AppConfig


class FakeClient:
    """메모리 추출(chat, json_mode)과 퇴고 스트림(stream_messages)만 흉내 낸다."""

    dry_run = False

    def __init__(self, *, summary: str = "에블린이 파혼을 선언했다.", fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def chat(self, prompt: str, system: str | None = None, **options: object) -> str:
        self.chat_calls.append({"prompt": prompt, "system": system, **options})
        if self.fail:
            raise RuntimeError("model down")
        return json.dumps(
            {
                "title": "파혼 선언",
                "summary": self.summary,
                "characters": ["에블린", "칼릭스"],
                "facts": ["파혼 서류가 황태자에게 전달됐다"],
                "open_clues": ["시한부 진단서의 출처"],
                "keywords": ["파혼", "서류"],
            },
            ensure_ascii=False,
        )

    def stream_messages(self, messages: list[dict[str, str]], **options: object):
        self.stream_calls.append({"messages": messages, **options})
        yield "### 파혼 선언\n\n"
        yield "고쳐 쓴 본문이다. "
        yield "에블린은 서류를 내밀었다."


DRAFT = (
    "### 1장 파혼 선언\n\n에블린은 파혼 서류를 던졌다. 칼릭스는 말이 없었다.\n\n"
    "### 2장 서고의 밤\n\n에블린은 봉인된 서고로 향했다. 문은 잠겨 있지 않았다."
)


class StoryEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.owner = self.store.create_user(username="editor", display_name="Editor", password="editor-pass-1")
        self.intruder = self.store.create_user(username="intruder", display_name="Intruder", password="intruder-pass-1")
        self.uid = str(self.owner["id"])
        self.story = self.store.create_story(
            self.uid,
            title="파혼 선언", genre="궁중 로맨스 판타지",
            premise="시한부 판정을 받은 악녀가 파혼을 선언한다.",
            world="황실과 귀족 가문이 예언으로 얽힌 제국",
            protagonist="에블린: 시한부 판정을 받은 공작가 영애",
            characters="칼릭스: 황태자",
            target_chars=10000,
            style_guide="1인칭 시점, 짧은 문장",
        )
        self.sid = str(self.story["id"])
        self.workspace = StoryWorkspace.for_story(self.config, self.sid, create=True)
        self.workspace.draft.write_text(DRAFT, encoding="utf-8")
        write_story_memories(
            self.workspace.memory,
            [
                StoryMemory(section_index=1, title="1장 파혼 선언", summary="옛 요약 하나"),
                StoryMemory(section_index=2, title="2장 서고의 밤", summary="옛 요약 둘"),
            ],
        )
        self.workspace.state.write_text(json.dumps({"total_chars": 1, "section_count": 2, "memory_count": 2}), encoding="utf-8")
        self.store.sync_story_progress(self.sid, len(DRAFT), 2)
        self.client = FakeClient()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    # ---- 읽기 ----------------------------------------------------------------------------
    def test_list_sections_and_memory(self) -> None:
        sections = list_sections(self.workspace)
        self.assertEqual([s.index for s in sections], [1, 2])
        self.assertEqual(sections[1].title, "2장 서고의 밤")
        self.assertIn("봉인된 서고", sections[1].body)
        self.assertTrue(sections[0].label.startswith("1장 · 1장 파혼 선언 · "))
        self.assertEqual(section_memory(self.workspace, 2).summary, "옛 요약 둘")
        self.assertIsNone(section_memory(self.workspace, 9))

    def test_normalize_edit_keeps_the_old_title_when_the_user_drops_it(self) -> None:
        self.assertEqual(normalize_edit("그냥 본문만 적었다.", 1, "옛 소제목"), "### 옛 소제목\n\n그냥 본문만 적었다.")
        self.assertEqual(normalize_edit("### 새 소제목\n\n본문.", 1, "옛 소제목"), "### 새 소제목\n\n본문.")
        self.assertEqual(normalize_edit("#새 제목\n본문.", 1, "옛"), "### 새 제목\n\n본문.")
        with self.assertRaises(StoryEditError):
            normalize_edit("### 제목만\n\n", 1, "옛")

    def test_user_prose_is_kept_verbatim_below_the_heading(self) -> None:
        # 모델 출력 정리기는 '제목:'·'#'·'N장' 으로 시작하는 줄을 지운다. 사람이 쓴 본문에서는 그러면 안 된다.
        text = "### 2장 심층 구역\n\n2장의 사진이 바닥에 떨어져 있었다.\n\"제목: 그건 네가 정해.\"\n#1 구역으로 가자.\n\n\n\n서윤은 줍지 않았다."
        self.assertEqual(
            normalize_edit(text, 2, "옛"),
            "### 2장 심층 구역\n\n2장의 사진이 바닥에 떨어져 있었다.\n\"제목: 그건 네가 정해.\"\n#1 구역으로 가자.\n\n서윤은 줍지 않았다.",
        )

    def test_model_output_goes_through_the_generator_cleaner(self) -> None:
        self.assertEqual(normalize_edit("소제목: 라벨 제목\n본문.", 1, "옛 소제목", from_model=True), "### 라벨 제목\n\n본문.")
        self.assertEqual(normalize_edit("2장. 심층 구역\n본문.", 2, "옛", from_model=True), "### 심층 구역\n\n본문.")
        self.assertEqual(normalize_edit("### 장면 3\n\n본문.", 3, "옛", from_model=True), "### 장면 3\n\n본문.")
        self.assertEqual(normalize_edit("```\n본문만.\n```", 1, "옛", from_model=True), "### 옛\n\n본문만.")

    # ---- 직접 수정 ------------------------------------------------------------------------
    def test_replace_section_rewrites_draft_memory_ledger_state_and_progress(self) -> None:
        result = replace_section(
            self.config, self.store, self.client, self.uid, self.sid, 2,
            "### 2장 서고의 밤\n\n에블린은 서고 문을 부쉈다. 안에는 예언서가 있었다.",
        )
        self.assertTrue(result["changed"])
        draft = self.workspace.draft.read_text(encoding="utf-8")
        self.assertIn("서고 문을 부쉈다", draft)
        self.assertNotIn("잠겨 있지 않았다", draft)
        self.assertIn("파혼 서류를 던졌다", draft, "the untouched section survives")
        memories = load_story_memories(self.workspace.memory)
        self.assertEqual([m.section_index for m in memories], [1, 2])
        self.assertEqual(memories[0].summary, "옛 요약 하나")
        self.assertEqual(memories[1].summary, "에블린이 파혼을 선언했다.")
        self.assertEqual(result["memory"].summary, "에블린이 파혼을 선언했다.")
        self.assertIn("서고 문을 부쉈다", self.client.chat_calls[0]["prompt"], "memory is extracted from the NEW prose")
        self.assertTrue(self.workspace.ledger.exists())
        state = json.loads(self.workspace.state.read_text(encoding="utf-8"))
        self.assertEqual(state["total_chars"], len(draft.strip()))
        story = self.store.get_owned_story(self.uid, self.sid)
        self.assertEqual(int(story["current_chars"]), len(draft.strip()))
        self.assertEqual(int(story["section_count"]), 2)

    def test_edits_keep_turn_count_and_completion_flag(self) -> None:
        """sync_story_files must not inherit delete-turn semantics (turns_completed -1, reopened ending)."""
        self.workspace.state.write_text(
            json.dumps({"total_chars": 1, "section_count": 2, "memory_count": 2, "turns_completed": 3, "novel_completed": True}),
            encoding="utf-8",
        )
        replace_section(self.config, self.store, self.client, self.uid, self.sid, 2, "### 2장 서고의 밤\n\n결말을 다듬었다.")
        update_section_memory(self.config, self.store, self.uid, self.sid, 2, summary="결말 요약")
        refresh_section_memory(self.config, self.store, self.client, self.uid, self.sid, 2)
        state = json.loads(self.workspace.state.read_text(encoding="utf-8"))
        self.assertEqual(state["turns_completed"], 3)
        self.assertTrue(state["novel_completed"])
        self.assertEqual(state["section_count"], 2)
        draft = self.workspace.draft.read_text(encoding="utf-8")
        self.assertEqual(state["total_chars"], len(draft.strip()))

    def test_replace_without_refresh_keeps_the_old_memory(self) -> None:
        replace_section(self.config, self.store, self.client, self.uid, self.sid, 1, "### 1장 파혼 선언\n\n다른 본문.",
                        refresh_memory=False)
        self.assertEqual(load_story_memories(self.workspace.memory)[0].summary, "옛 요약 하나")
        self.assertEqual(self.client.chat_calls, [])

    def test_unchanged_text_reports_no_change_and_writes_nothing(self) -> None:
        sections = list_sections(self.workspace)
        before = self.store.get_owned_story(self.uid, self.sid)["updated_at"]
        memory_mtime = self.workspace.memory.stat().st_mtime_ns
        result = replace_section(self.config, self.store, self.client, self.uid, self.sid, 1, sections[0].text,
                                 refresh_memory=False)
        self.assertFalse(result["changed"])
        self.assertEqual(result["memory"].summary, "옛 요약 하나")
        self.assertEqual(self.store.get_owned_story(self.uid, self.sid)["updated_at"], before)
        self.assertEqual(self.workspace.memory.stat().st_mtime_ns, memory_mtime)

    def test_model_failure_falls_back_to_a_mechanical_summary(self) -> None:
        result = replace_section(self.config, self.store, FakeClient(fail=True), self.uid, self.sid, 1,
                                 "### 1장 파혼 선언\n\n에블린은 서류를 찢었다. 칼릭스가 웃었다.")
        self.assertIn("서류를 찢었다", result["memory"].summary)
        self.assertIn("에블린", result["memory"].characters)

    def test_missing_section_and_empty_body_are_refused(self) -> None:
        with self.assertRaises(StoryEditError):
            replace_section(self.config, self.store, self.client, self.uid, self.sid, 3, "### x\n\n본문")
        with self.assertRaises(StoryEditError):
            replace_section(self.config, self.store, self.client, self.uid, self.sid, 1, "   ")

    def test_other_accounts_and_busy_stories_are_refused(self) -> None:
        with self.assertRaises(AuthorizationError):
            replace_section(self.config, self.store, self.client, str(self.intruder["id"]), self.sid, 1, "### a\n\nb")
        self.store.enqueue_job(self.uid, self.sid, instruction="다음 장면.", requested_chars=2000, creativity_profile="balanced")
        with self.assertRaises(StoryBusyError):
            replace_section(self.config, self.store, self.client, self.uid, self.sid, 1, "### a\n\nb")
        with self.assertRaises(StoryBusyError):
            refresh_section_memory(self.config, self.store, self.client, self.uid, self.sid, 1)
        with self.assertRaises(StoryBusyError):
            update_section_memory(self.config, self.store, self.uid, self.sid, 1, summary="x")
        self.assertIn("잠겨 있지 않았다", self.workspace.draft.read_text(encoding="utf-8"))

    def test_memories_are_padded_when_the_file_is_short(self) -> None:
        write_story_memories(self.workspace.memory, [StoryMemory(section_index=1, title="t", summary="하나만")])
        replace_section(self.config, self.store, self.client, self.uid, self.sid, 1, "### 1장 파혼 선언\n\n바뀐 본문.")
        memories = load_story_memories(self.workspace.memory)
        self.assertEqual([m.section_index for m in memories], [1, 2])
        self.assertIn("봉인된 서고", memories[1].summary, "the padded memory is built from section 2's prose")

    # ---- 메모리만 ------------------------------------------------------------------------
    def test_refresh_and_manual_memory_update(self) -> None:
        refreshed = refresh_section_memory(self.config, self.store, self.client, self.uid, self.sid, 2)
        self.assertEqual(refreshed.summary, "에블린이 파혼을 선언했다.")
        self.assertIn("봉인된 서고", self.client.chat_calls[-1]["prompt"])
        manual = update_section_memory(
            self.config, self.store, self.uid, self.sid, 2,
            summary="  에블린이   서고에서 예언서를 찾았다. ", facts="예언서는 황실 서고에 있다\n\n문은 열려 있었다", open_clues="누가 문을 열어 두었나",
        )
        self.assertEqual(manual.summary, "에블린이 서고에서 예언서를 찾았다.")
        self.assertEqual(manual.facts, ["예언서는 황실 서고에 있다", "문은 열려 있었다"])
        self.assertEqual(manual.open_clues, ["누가 문을 열어 두었나"])
        self.assertEqual(section_memory(self.workspace, 2).summary, "에블린이 서고에서 예언서를 찾았다.")
        with self.assertRaises(StoryEditError):
            update_section_memory(self.config, self.store, self.uid, self.sid, 2, summary="   ")

    # ---- AI 퇴고 ----------------------------------------------------------------------------
    def test_rewrite_prompt_carries_settings_prose_and_request(self) -> None:
        messages = rewrite_messages(self.story, "### 1장\n\n원문.", "더 긴장감 있게")
        self.assertEqual(messages[0]["role"], "system")
        user = messages[1]["content"]
        self.assertIn("[집필 지침", user)
        self.assertIn("1인칭 시점, 짧은 문장", user)
        self.assertNotIn("집필 지침: ", world_sheet(self.story), "the world sheet stays setting-only")
        self.assertIn("에블린: 시한부 판정을 받은 공작가 영애", user)
        self.assertIn("원문.", user)
        self.assertIn("긴장감을 높인다", user, "quick labels expand to concrete directives")
        self.assertIn("원문 문장을 그대로 옮겨 적지 않는다", user, "style requests forbid verbatim copying")
        self.assertTrue(QUICK_REWRITES)

    def test_rewrite_plan_modes_and_temperatures(self) -> None:
        self.assertEqual(rewrite_plan("오탈자와 어색한 문장만 고쳐")[1], "minimal")
        self.assertEqual(rewrite_plan("더 긴장감 있게")[1], "style")
        self.assertEqual(rewrite_plan("더 긴장감 있게 / 민재의 대사를 거칠게")[1], "style")
        self.assertEqual(rewrite_plan("민재의 대사를 거칠게. 마지막 문단은 그대로.")[1], "targeted")
        self.assertEqual(rewrite_plan("맞춤법만 봐줘")[1], "minimal")
        self.assertEqual(rewrite_plan("")[0], ["문장을 자연스럽게 다듬는다."])
        self.assertEqual(rewrite_temperature("오탈자와 어색한 문장만 고쳐"), 0.4)
        self.assertEqual(rewrite_temperature("더 긴장감 있게"), 0.8)
        self.assertEqual(rewrite_temperature("민재의 대사를 거칠게"), 0.6)
        targeted = rewrite_messages(self.story, "### 1장\n\n원문.", "민재의 대사를 거칠게")[1]["content"]
        self.assertIn("나머지 문장은 원문을 유지한다", targeted)
        minimal = rewrite_messages(self.story, "### 1장\n\n원문.", "오탈자와 어색한 문장만 고쳐")[1]["content"]
        self.assertIn("고칠 곳이 아닌 문장은 원문 그대로 둔다", minimal)

    def test_plan_edits_parses_the_model_plan_and_feeds_the_rewrite(self) -> None:
        class PlanningClient(FakeClient):
            def chat(self, prompt: str, system: str | None = None, **options: object) -> str:
                self.chat_calls.append({"prompt": prompt, "system": system, **options})
                return ('{"edits": [{"where": "서류를 던졌다", "how": "손이 떨리는 묘사를 더한다"}, '
                        '{"where": "", "how": "버림"}, {"where": "말이 없었다", "how": "짧은 대사로 바꾼다"}]}')

        client = PlanningClient()
        plan = plan_edits(client, self.config, self.story, "### 1장\n\n원문.", "더 긴장감 있게")
        self.assertEqual([e["where"] for e in plan], ["서류를 던졌다", "말이 없었다"])
        self.assertTrue(client.chat_calls[0]["json_mode"])
        user = rewrite_messages(self.story, "### 1장\n\n원문.", "더 긴장감 있게", plan)[1]["content"]
        self.assertIn("[수정 계획", user)
        self.assertIn("'서류를 던졌다' → 손이 떨리는 묘사를 더한다", user)
        self.assertEqual(plan_edits(client, self.config, self.story, "x", "오탈자와 어색한 문장만 고쳐"), [])
        self.assertEqual(plan_edits(FakeClient(fail=True), self.config, self.story, "x", "더 긴장감 있게"), [])
        self.assertNotIn("[수정 계획", rewrite_messages(self.story, "### 1장\n\n원문.", "더 긴장감 있게")[1]["content"])

    def test_length_guard_flags_out_of_range_proposals(self) -> None:
        original = "### 1장" + chr(10) * 2 + "가" * 1000
        self.assertEqual(length_bounds("더 긴장감 있게"), (0.7, 1.4))
        self.assertEqual(length_bounds("문장을 짧고 담백하게"), (0.45, 1.4))
        self.assertEqual(length_bounds("감각 묘사를 풍부하게"), (0.7, 1.8))
        self.assertIn("너무 짧았다", retry_note_for(original, "가" * 400, "더 긴장감 있게"))
        self.assertEqual(retry_note_for(original, "가" * 500, "문장을 짧고 담백하게"), "", "shortening was asked for")
        self.assertIn("너무 길었다", retry_note_for(original, "가" * 1600, "더 긴장감 있게"))
        self.assertEqual(retry_note_for(original, "가" * 950, "더 긴장감 있게"), "")
        user = rewrite_messages(self.story, original, "더 긴장감 있게", None, "분량이 너무 짧았다.")[1]["content"]
        self.assertIn("[이전 시도의 문제", user)
        self.assertTrue(user.rstrip().endswith("- 분량이 너무 짧았다."))

    def test_stream_rewrite_uses_the_base_model_with_strict_korean_filter(self) -> None:
        chunks = list(stream_rewrite(self.client, self.config, self.story, "### 1장\n\n원문.", "짧게"))
        self.assertEqual("".join(chunks), "### 파혼 선언\n\n고쳐 쓴 본문이다. 에블린은 서류를 내밀었다.")
        call = self.client.stream_calls[0]
        self.assertIsNone(call["adapter"])
        self.assertEqual(call["korean_filter"], "strict")
        self.assertGreaterEqual(call["max_tokens"], 800)


class StorySheetTests(unittest.TestCase):
    def test_world_sheet_is_setting_only_and_style_guide_is_separate(self) -> None:
        story = {"genre": "SF", "premise": "소재", "world": "세계", "style_guide": ""}
        self.assertEqual(style_guide(story), "")
        story["style_guide"] = "  3인칭 관찰자 시점 "
        self.assertEqual(style_guide(story), "3인칭 관찰자 시점")
        self.assertNotIn("집필 지침", world_sheet(story))
        self.assertTrue(world_sheet(story).endswith("세계관: 세계"))

    def test_section_prompt_renders_the_guide_as_its_own_block(self) -> None:
        from src.generation.longform import WRITE_ROLE_MARKER, _canon_prefix, _section_prompt, _task_block

        canon = dict(world="w", characters="c", outline_text="o", memory_context="m", consumed_context="x", tail="t")
        task = _task_block(section_index=2, section_role="r", function_name="f", function_rule="fr", target_chars=1800,
                           completion_rule="cr", instruction="i")
        plain = _section_prompt(_canon_prefix(**canon), task)
        self.assertNotIn("[집필 지침", plain)
        guided_prefix = _canon_prefix(**canon, style_guide="1인칭, 짧은 문장")
        # 집필 지침은 공유 접두사의 마지막 블록이다 (설계·검토·기록 호출도 같은 규칙을 본다).
        self.assertTrue(guided_prefix.endswith("[집필 지침 — 작가가 정한 문체·시점·금기]" + chr(10) + "1인칭, 짧은 문장" + chr(10)))
        guided = _section_prompt(guided_prefix, task)
        self.assertIn("1인칭, 짧은 문장" + chr(10) + "[이번 장(2장)의 과제]", guided)
        self.assertIn(WRITE_ROLE_MARKER, guided)
        self.assertLess(guided.index("[집필 지침"), guided.index(WRITE_ROLE_MARKER))

    def test_character_sheet_rules(self) -> None:
        self.assertEqual(character_sheet({"protagonist": "서윤", "characters": ""}), "서윤: 작품의 주인공")
        self.assertEqual(character_sheet({"protagonist": "서윤: 복원가", "characters": ""}), "서윤: 복원가")
        self.assertEqual(character_sheet({"protagonist": "서윤: 복원가", "characters": "민재: 연구원"}), "서윤: 복원가\n민재: 연구원")

    def test_style_guide_round_trips_through_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            store = ConsumerStore(config)
            user = store.create_user(username="guide", display_name="G", password="guide-pass-1")
            story = store.create_story(
                str(user["id"]), title="t", genre="g", premise="p", world="w", protagonist="x",
                target_chars=10000, style_guide="  담백한 문체\r\n대화 위주  ",
            )
            self.assertEqual(story["style_guide"], "담백한 문체\n대화 위주")
            updated = store.update_owned_story(
                str(user["id"]), str(story["id"]), title="t", genre="g", premise="p", world="w", protagonist="x",
                characters="", target_chars=10000, research_consent=False, style_guide="x" * 3000,
            )
            self.assertEqual(len(updated["style_guide"]), 2000)
            cleared = store.update_owned_story(
                str(user["id"]), str(story["id"]), title="t", genre="g", premise="p", world="w", protagonist="x",
                characters="", target_chars=10000, research_consent=False,
            )
            self.assertEqual(cleared["style_guide"], "")


if __name__ == "__main__":
    unittest.main()
