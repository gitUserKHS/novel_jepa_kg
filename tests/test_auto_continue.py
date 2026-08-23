"""자동 이어쓰기: 워커가 턴을 마치면 다음 턴을 스스로 넣고, 지시는 모델이 정해 작업 기록에 남긴다."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.service.auto_continue import AUTO_FALLBACK_DIRECTION, AUTO_PLACEHOLDER, clean_direction, propose_next_direction
from src.service.consumer_store import JOB_ORIGIN_AUTO, JOB_ORIGIN_USER, JOB_SUCCEEDED, AuthorizationError, ConsumerStore
from src.service.story_workspace import StoryWorkspace
from src.service.worker import ConsumerWorker
from src.utils.config import AppConfig
from tests.test_consumer_worker import FakeGenerator, fake_manifest


class PlannerClient:
    """전개 제안 호출만 흉내 낸다."""

    def __init__(self, direction: str = "서윤이 좌표를 따라 심층 구역으로 내려가게 해줘.", fail: bool = False) -> None:
        self.direction = direction
        self.fail = fail
        self.prompts: list[str] = []

    def chat(self, prompt: str, system: str | None = None, **_options: Any) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("model down")
        return self.direction


class AutoContinueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.store = ConsumerStore(self.config)
        self.user = self.store.create_user(username="auto_user", display_name="Auto", password="auto-password")
        self.uid = str(self.user["id"])
        self.story = self.store.create_story(
            self.uid, title="자동 연재", genre="SF 미스터리", premise="기억 도난 사건",
            world="기억이 거래되는 도시", protagonist="서윤: 기록 복원가", characters="민재: 연구원",
            target_chars=15000, research_consent=False,
        )
        self.sid = str(self.story["id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _worker(self, planner: PlannerClient | None = None) -> ConsumerWorker:
        planner = planner or PlannerClient()
        return ConsumerWorker(
            self.config, store=self.store, worker_id="auto-worker",
            client_factory=lambda _config: planner, generator=FakeGenerator(), manifest_loader=fake_manifest,
        )

    def _jobs(self) -> list[dict[str, Any]]:
        return sorted(self.store.list_jobs(self.sid), key=lambda job: int(job["id"]))

    # ---- store ------------------------------------------------------------------------------
    def test_store_toggle_origin_and_streak(self) -> None:
        self.assertFalse(self.store.get_owned_story(self.uid, self.sid)["auto_continue"])
        on = self.store.set_auto_continue(self.uid, self.sid, True)
        self.assertTrue(on["auto_continue"])
        with self.assertRaises(AuthorizationError):
            self.store.set_auto_continue("nobody", self.sid, False)
        user_job = self.store.enqueue_job(self.uid, self.sid, instruction="사람이 적은 전개", creativity_profile="balanced",
                                          requested_chars=2000)
        self.assertEqual(user_job["origin"], JOB_ORIGIN_USER)
        self.assertEqual(self.store.consecutive_auto_jobs(self.sid), 0)
        with self.assertRaises(ValueError):
            self.store.enqueue_job(self.uid, self.sid, instruction="x", creativity_profile="balanced",
                                   requested_chars=2000, origin="robot")
        self.store.set_job_instruction(int(user_job["id"]), "  바뀐   지시  ")
        self.assertEqual(self.store.get_job(int(user_job["id"]))["instruction"], "바뀐 지시")
        self.store.set_job_instruction(int(user_job["id"]), "   ")
        self.assertEqual(self.store.get_job(int(user_job["id"]))["instruction"], "바뀐 지시", "blank is ignored")

    # ---- worker chain -----------------------------------------------------------------------
    def test_auto_turns_chain_until_the_novel_completes(self) -> None:
        self.store.set_auto_continue(self.uid, self.sid, True)
        planner = PlannerClient()
        worker = self._worker(planner)
        first = self.store.enqueue_job(self.uid, self.sid, instruction=AUTO_PLACEHOLDER, creativity_profile="bold",
                                       requested_chars=5000, origin=JOB_ORIGIN_AUTO)
        turns = 0
        while worker.process_one():
            turns += 1
            self.assertLess(turns, 20, "the chain must end on its own")
        jobs = self._jobs()
        self.assertEqual(turns, math.ceil(15000 / 5000))
        self.assertEqual(len(jobs), turns)
        self.assertTrue(all(job["status"] == JOB_SUCCEEDED for job in jobs))
        self.assertTrue(all(job["origin"] == JOB_ORIGIN_AUTO for job in jobs))
        self.assertTrue(all(job["instruction"] == planner.direction for job in jobs), [j["instruction"] for j in jobs])
        self.assertEqual(int(jobs[0]["id"]), int(first["id"]))
        self.assertTrue(all(job["creativity_profile"] == "bold" and int(job["requested_chars"]) == 5000 for job in jobs),
                        "auto turns reuse the settings of the turn that started the chain")
        story = self.store.get_owned_story(self.uid, self.sid)
        self.assertIsNotNone(story["completed_at"], "the chain wrote the novel to its ending")
        self.assertIsNone(self.store.owned_outstanding_job(self.uid, self.sid), "nothing queued after completion")
        self.assertEqual(self.store.consecutive_auto_jobs(self.sid), turns)
        # 제안 프롬프트는 설정·인물·줄거리를 담는다 (두 번째 턴부터 메모리가 있다)
        self.assertIn("서윤: 기록 복원가", planner.prompts[0])
        self.assertIn("장이 맡을 단계", planner.prompts[0])
        self.assertIn("지금까지의 줄거리", planner.prompts[1])

    def test_turn_cap_switches_auto_off(self) -> None:
        self.config.consumer.auto_continue_max_turns = 2
        self.store.set_auto_continue(self.uid, self.sid, True)
        worker = self._worker()
        self.store.enqueue_job(self.uid, self.sid, instruction="첫 전개", creativity_profile="balanced",
                               requested_chars=2000)
        processed = 0
        while worker.process_one():
            processed += 1
            self.assertLess(processed, 10)
        jobs = self._jobs()
        self.assertEqual([job["origin"] for job in jobs], [JOB_ORIGIN_USER, JOB_ORIGIN_AUTO, JOB_ORIGIN_AUTO])
        self.assertFalse(self.store.get_owned_story(self.uid, self.sid)["auto_continue"], "cap reached -> switched off")
        self.assertIsNone(self.store.owned_outstanding_job(self.uid, self.sid))
        self.assertIsNone(self.store.get_owned_story(self.uid, self.sid)["completed_at"], "far from the target")

    def test_auto_off_means_no_chain_and_user_turns_reset_the_streak(self) -> None:
        worker = self._worker()
        self.store.enqueue_job(self.uid, self.sid, instruction="첫 전개", creativity_profile="balanced", requested_chars=2000)
        self.assertTrue(worker.process_one())
        self.assertFalse(worker.process_one(), "auto off: nothing else is queued")
        self.assertEqual(len(self._jobs()), 1)
        # 자동 턴 둘 뒤에 사람이 넣은 턴이 오면 연속 수는 0 부터
        self.store.enqueue_job(self.uid, self.sid, instruction=AUTO_PLACEHOLDER, creativity_profile="balanced",
                               requested_chars=2000, origin=JOB_ORIGIN_AUTO)
        self.assertTrue(worker.process_one())
        self.assertEqual(self.store.consecutive_auto_jobs(self.sid), 1)
        self.store.enqueue_job(self.uid, self.sid, instruction="사람 전개", creativity_profile="balanced", requested_chars=2000)
        self.assertEqual(self.store.consecutive_auto_jobs(self.sid), 0)

    def test_planner_failure_falls_back_to_the_default_direction(self) -> None:
        worker = self._worker(PlannerClient(fail=True))
        job = self.store.enqueue_job(self.uid, self.sid, instruction=AUTO_PLACEHOLDER, creativity_profile="balanced",
                                     requested_chars=2000, origin=JOB_ORIGIN_AUTO)
        self.assertTrue(worker.process_one())
        done = self.store.get_job(int(job["id"]))
        self.assertEqual(done["status"], JOB_SUCCEEDED)
        self.assertEqual(done["instruction"], AUTO_FALLBACK_DIRECTION)

    # ---- direction proposal -------------------------------------------------------------------
    def test_propose_direction_reads_outline_and_memory(self) -> None:
        workspace = StoryWorkspace.for_story(self.config, self.sid, create=True)
        workspace.draft.write_text("### 1장\n\n서윤은 장치를 켰다.", encoding="utf-8")
        workspace.memory.write_text(json.dumps({"section_index": 1, "title": "1장", "summary": "서윤이 장치를 켰다.",
                                                "open_clues": ["심층 구역 좌표"]}, ensure_ascii=False) + "\n", encoding="utf-8")
        workspace.outline.write_text(json.dumps({
            "premise": "p", "ending_intent": "e",
            "beats": [{"beat_id": i, "phase": f"{i}막", "purpose": f"비트 {i} 목적", "required_change": f"변화 {i}"} for i in (1, 2, 3)],
        }, ensure_ascii=False), encoding="utf-8")
        planner = PlannerClient('"지시: 민재가 서윤을 막아서게 해줘. 좌표의 출처를 캐묻는다. 셋째 문장은 버린다."')
        direction = propose_next_direction(planner, self.config, self.store.get_story(self.sid), workspace)
        self.assertEqual(direction, "민재가 서윤을 막아서게 해줘. 좌표의 출처를 캐묻는다.")
        prompt = planner.prompts[0]
        self.assertIn("2장이 맡을 단계", prompt)
        self.assertIn("심층 구역 좌표", prompt)
        self.assertIn("서윤이 장치를 켰다", prompt)
        self.assertIn("남은 분량", prompt)
        self.assertNotIn("이미 시킨 전개", prompt, "nothing asked before -> no block")
        self.assertEqual(propose_next_direction(PlannerClient(fail=True), self.config, self.store.get_story(self.sid), workspace), "")

    def test_previous_directions_are_listed_so_the_planner_does_not_repeat_them(self) -> None:
        workspace = StoryWorkspace.for_story(self.config, self.sid, create=True)
        planner = PlannerClient("선아가 유리의 작업실을 찾아가게 해줘.")
        propose_next_direction(planner, self.config, self.store.get_story(self.sid), workspace,
                               previous_directions=["첫 만남을 그려줘", AUTO_PLACEHOLDER, "유리가 비밀을 털어놓게 해줘.", "  "])
        prompt = planner.prompts[0]
        self.assertIn("[이미 시킨 전개 — 같은 소재·같은 사건을 다시 고르지 않는다]", prompt)
        self.assertIn("- 첫 만남을 그려줘", prompt)
        self.assertIn("- 유리가 비밀을 털어놓게 해줘.", prompt)
        self.assertNotIn(AUTO_PLACEHOLDER, prompt, "placeholders and blanks are not 'asked' directions")

    def test_worker_passes_earlier_turns_to_the_planner(self) -> None:
        planner = PlannerClient()
        worker = self._worker(planner)
        self.store.enqueue_job(self.uid, self.sid, instruction="유리가 비밀을 털어놓게 해줘.", creativity_profile="balanced",
                               requested_chars=2000)
        self.assertTrue(worker.process_one())
        self.store.enqueue_job(self.uid, self.sid, instruction=AUTO_PLACEHOLDER, creativity_profile="balanced",
                               requested_chars=2000, origin=JOB_ORIGIN_AUTO)
        self.assertTrue(worker.process_one())
        self.assertIn("- 유리가 비밀을 털어놓게 해줘.", planner.prompts[-1])

    def test_clean_direction(self) -> None:
        self.assertEqual(clean_direction("```\n박 노인이 털어놓게 해줘.\n```"), "박 노인이 털어놓게 해줘.")
        self.assertEqual(clean_direction("  “서윤이   달린다.”  "), "서윤이 달린다.")
        self.assertEqual(clean_direction("짧"), "")
        self.assertEqual(clean_direction(""), "")
        self.assertLessEqual(len(clean_direction("가" * 500 + ".")), 200)


if __name__ == "__main__":
    unittest.main()
