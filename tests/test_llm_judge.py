from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.llm_judge import JUDGE_SCORE_KEYS, judge_excerpt, run_llm_judge
from src.evaluation.report import evaluate_and_write_report, evaluate_outputs
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig
from src.utils.paths import resolve_path


PREVIOUS_SCENE = "하린은 언니의 기억 조각이 불법 경매에 올라왔다는 사실을 알게 된다."
GENERATED = (
    "### 경매장\n\n"
    "하린은 위조된 초대장을 쥐고 지하 경매장으로 내려갔다. "
    "낯선 향이 기억 조각 보관함에서 흘러나왔고, 그 향은 언니의 방에서 맡았던 것과 같았다.\n\n"
    "### 낙찰\n\n"
    "호가가 오를수록 하린의 손끝이 차가워졌다. 그녀는 언니의 목소리를 되찾기 위해 "
    "마지막 남은 신뢰 점수를 전부 걸었다."
)


class ScriptedClient:
    """Minimal judge/embed client double for evaluation tests."""

    dry_run = False

    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_json_mode: bool | None = None
        self.chat_calls = 0

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        stream_callback=None,
    ) -> str:
        self.chat_calls += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_json_mode = json_mode
        if self.error is not None:
            raise self.error
        return self.response

    def embed(self, texts: list[str], *, unload_chat: bool | None = None) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vector = rng.normal(size=64)
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors).astype("float32")


class LlmJudgeTests(unittest.TestCase):
    def test_it_parses_scores_wrapped_in_extra_prose(self) -> None:
        payload = {
            "plausibility": 8,
            "creativity": 9,
            "hallucination_control": 7,
            "consistency": 8,
            "immersion": 7,
            "useful_hallucinations": ["낯선 향이 언니의 방과 연결된다."],
            "harmful_hallucinations": [],
            "verdict": "설정을 지키면서 새 단서를 심었다.",
        }
        client = ScriptedClient(response=f"채점 결과입니다.\n{json.dumps(payload, ensure_ascii=False)}\n이상입니다.")

        result = run_llm_judge(client, PREVIOUS_SCENE, GENERATED)

        self.assertTrue(result["available"])
        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(set(result["scores"]), set(JUDGE_SCORE_KEYS))
        self.assertAlmostEqual(result["overall"], 39 / 50)
        self.assertEqual(result["useful_hallucinations"], payload["useful_hallucinations"])
        self.assertEqual(result["verdict"], payload["verdict"])
        self.assertTrue(client.last_json_mode)
        self.assertIn(PREVIOUS_SCENE, client.last_prompt)

    def test_scores_are_clamped_and_bad_values_dropped(self) -> None:
        payload = {
            "plausibility": 15,
            "creativity": -3,
            "hallucination_control": "높음",
            "consistency": 8,
            "immersion": None,
            "useful_hallucinations": "단일 문자열도 목록으로 감싼다.",
            "harmful_hallucinations": ["a", "", "b", "c", "d", "e", "f"],
            "verdict": "총평",
        }
        client = ScriptedClient(response=json.dumps(payload, ensure_ascii=False))

        result = run_llm_judge(client, PREVIOUS_SCENE, GENERATED)

        self.assertTrue(result["available"])
        self.assertEqual(result["scores"]["plausibility"], 10.0)
        self.assertEqual(result["scores"]["creativity"], 1.0)
        self.assertNotIn("hallucination_control", result["scores"])
        self.assertNotIn("immersion", result["scores"])
        self.assertEqual(result["useful_hallucinations"], ["단일 문자열도 목록으로 감싼다."])
        self.assertEqual(len(result["harmful_hallucinations"]), 5)

    def test_backend_failure_downgrades_to_unavailable(self) -> None:
        client = ScriptedClient(error=RuntimeError("Ollama /api/chat failed"))

        result = run_llm_judge(client, PREVIOUS_SCENE, GENERATED)

        self.assertFalse(result["available"])
        self.assertIn("Ollama /api/chat failed", result["error"])
        self.assertEqual(result["scores"], {})

    def test_unparseable_output_downgrades_to_unavailable(self) -> None:
        client = ScriptedClient(response="점수를 드리기 어렵습니다.")

        result = run_llm_judge(client, PREVIOUS_SCENE, GENERATED)

        self.assertFalse(result["available"])

    def test_empty_generated_text_is_unavailable_without_a_chat_call(self) -> None:
        client = ScriptedClient(response="{}")

        result = run_llm_judge(client, PREVIOUS_SCENE, "   ")

        self.assertFalse(result["available"])
        self.assertEqual(client.chat_calls, 0)

    def test_dry_run_returns_a_stable_stub(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", chat_model="x", embed_model="y", dry_run=True)

        result = run_llm_judge(client, PREVIOUS_SCENE, GENERATED)

        self.assertTrue(result["available"])
        self.assertEqual(result["backend"], "dry-run")
        self.assertEqual(set(result["scores"]), set(JUDGE_SCORE_KEYS))

    def test_long_prose_is_excerpted_head_and_tail(self) -> None:
        text = "시작" + ("가" * 5000) + "끝"

        excerpt = judge_excerpt(text, max_chars=1000)

        self.assertLess(len(excerpt), 1100)
        self.assertTrue(excerpt.startswith("시작"))
        self.assertTrue(excerpt.endswith("끝"))
        self.assertIn("[...중략...]", excerpt)


class JudgeReportIntegrationTests(unittest.TestCase):
    def test_evaluate_outputs_attaches_judge_rows_without_changing_ranking(self) -> None:
        config = AppConfig()
        payload = {key: 8 for key in JUDGE_SCORE_KEYS}
        client = ScriptedClient(response=json.dumps(payload))
        outputs = {"creative_jepa": GENERATED}

        config.evaluation.use_llm_judge = False
        baseline = evaluate_outputs(config, client, PREVIOUS_SCENE, outputs)
        config.evaluation.use_llm_judge = True
        judged = evaluate_outputs(config, client, PREVIOUS_SCENE, outputs, world="기억 거래가 합법인 근미래 서울")

        self.assertNotIn("llm_judge", baseline["modes"]["creative_jepa"])
        self.assertTrue(judged["modes"]["creative_jepa"]["llm_judge"]["available"])
        self.assertEqual(
            baseline["modes"]["creative_jepa"]["overall_score"],
            judged["modes"]["creative_jepa"]["overall_score"],
        )
        self.assertEqual(baseline["ranking"], judged["ranking"])
        self.assertIn("기억 거래가 합법인 근미래 서울", client.last_prompt)

    def test_report_includes_judge_section_and_survives_judge_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            config.evaluation.use_llm_judge = True
            client = ScriptedClient(error=RuntimeError("runner stopped"))

            path = evaluate_and_write_report(config, client, PREVIOUS_SCENE, {"creative_jepa": GENERATED})

            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("## LLM Judge", content)
            self.assertIn("unavailable: runner stopped", content)

    def test_report_marks_the_judge_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(output_root=temporary)
            client = ScriptedClient(response="{}")

            path = evaluate_and_write_report(config, client, PREVIOUS_SCENE, {"creative_jepa": GENERATED})

            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("## LLM Judge", content)
            self.assertIn("use_llm_judge: false", content)
            self.assertEqual(client.chat_calls, 0)
            self.assertTrue(Path(resolve_path(config, config.evaluation.report_dir)).exists())


if __name__ == "__main__":
    unittest.main()
