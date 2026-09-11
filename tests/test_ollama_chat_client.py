from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.llm.common import JSON_MODE_HINT, LocalLLMError
from src.llm.ollama_chat import OllamaChatClient, _iter_ndjson, model_installed

MODEL = "hf.co/team/Gemma4-26B-A4B:Q4_K_M"


def _line(**payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


class FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200, text: str = "") -> None:
        self.lines = lines
        self.status_code = status_code
        self.text = text

    def iter_lines(self, decode_unicode: bool = True):  # noqa: ANN201
        return iter(self.lines)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return json.loads(self.text)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class NDJSONParserTests(unittest.TestCase):
    def test_content_pieces_thinking_ignored_and_final_payload(self) -> None:
        lines = [
            _line(message={"role": "assistant", "content": "", "thinking": "Korean"}, done=False),
            _line(message={"role": "assistant", "content": "안녕"}, done=False),
            "",
            _line(message={"role": "assistant", "content": "하세요"}, done=False),
            _line(message={"role": "assistant", "content": ""}, done=True, done_reason="stop",
                  prompt_eval_count=12, eval_count=2),
        ]
        pieces = list(_iter_ndjson(FakeResponse(lines)))
        self.assertEqual([piece for piece, final in pieces if final is None], ["안녕", "하세요"])
        self.assertEqual(pieces[-1][1]["done_reason"], "stop")

    def test_error_line_raises(self) -> None:
        with self.assertRaisesRegex(LocalLLMError, "context"):
            list(_iter_ndjson(FakeResponse([_line(error="context length exceeded")])))

    def test_garbage_lines_are_skipped(self) -> None:
        lines = ["not json", _line(message={"role": "assistant", "content": "가"}, done=False), "[]",
                 _line(done=True)]
        pieces = list(_iter_ndjson(FakeResponse(lines)))
        self.assertEqual(pieces[0], ("가", None))
        self.assertIsNotNone(pieces[-1][1])


class RequestBodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OllamaChatClient("http://127.0.0.1:9", MODEL, num_ctx=8192, keep_alive="15m", top_k=40,
                                       retry_attempts=0)

    def test_body_carries_think_false_context_window_and_sampling(self) -> None:
        body = self.client._build_body(
            [{"role": "user", "content": "x"}], temperature=0.5, max_tokens=300, json_mode=True, seed=7,
            stop=["###"], top_p=None, top_k=None, repetition_penalty=None, min_p=0.0,
        )
        self.assertIs(body["think"], False, "Gemma4 otherwise answers only in the thinking field")
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["keep_alive"], "15m")
        self.assertTrue(body["stream"])
        options = body["options"]
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 300)
        self.assertEqual(options["seed"], 7)
        self.assertEqual(options["stop"], ["###"])
        self.assertEqual(options["repeat_penalty"], 1.05)
        self.assertEqual(options["top_k"], 40)
        self.assertNotIn("min_p", options)

    def test_json_hint_goes_to_the_user_turn_not_the_system_prompt(self) -> None:
        messages = [{"role": "system", "content": "시스템"}, {"role": "user", "content": "질문"}]
        payload = OllamaChatClient._json_mode_messages(messages)
        self.assertEqual(payload[0]["content"], "시스템", "the shared prefix must stay cacheable")
        self.assertTrue(payload[1]["content"].endswith(JSON_MODE_HINT))
        self.assertEqual(messages[1]["content"], "질문", "the caller's messages are not mutated")

    def test_local_server_only_options_are_ignored(self) -> None:
        client = OllamaChatClient("http://127.0.0.1:9", dry_run=True)
        text = client.chat("다음 장면", korean_filter="strict", dry_multiplier=0.8, adapter=None, no_repeat_ngram_size=16)
        self.assertIn("서윤", text)


class StreamingTests(unittest.TestCase):
    def test_chat_joins_pieces_and_records_usage(self) -> None:
        lines = [
            _line(message={"role": "assistant", "content": "안녕"}, done=False),
            _line(message={"role": "assistant", "content": "하세요"}, done=False),
            _line(message={"role": "assistant", "content": ""}, done=True, done_reason="stop",
                  prompt_eval_count=100, eval_count=5, prompt_eval_duration=2_000_000_000, eval_duration=1_000_000_000),
        ]
        client = OllamaChatClient("http://127.0.0.1:9", MODEL, num_ctx=4096, retry_attempts=0)
        with patch("src.llm.ollama_chat.requests.post", return_value=FakeResponse(lines)) as post:
            received: list[str] = []
            text = client.chat("프롬프트", system="작가", stream_callback=received.append, max_tokens=200)
        self.assertEqual(text, "안녕하세요")
        self.assertEqual("".join(received), text)
        self.assertEqual(client.last_usage["prompt_tokens"], 100)
        self.assertEqual(client.last_usage["output_tokens"], 5)
        self.assertEqual(client.last_usage["prompt_seconds"], 2.0)
        self.assertFalse(client.last_usage["truncated"])
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "작가"})
        self.assertEqual(sent["model"], MODEL)
        self.assertIs(sent["think"], False)

    def test_truncation_is_flagged_when_the_prompt_fills_the_window(self) -> None:
        client = OllamaChatClient("http://127.0.0.1:9", MODEL, num_ctx=4096)
        usage = client._record_usage({"prompt_eval_count": 3900, "eval_count": 10, "done_reason": "stop"}, 300)
        self.assertTrue(usage["truncated"])
        fine = client._record_usage({"prompt_eval_count": 2000, "eval_count": 10}, 300)
        self.assertFalse(fine["truncated"])

    def test_http_error_raises_local_llm_error(self) -> None:
        client = OllamaChatClient("http://127.0.0.1:9", MODEL, retry_attempts=0, retry_backoff_sec=0)
        with patch("src.llm.ollama_chat.requests.post", return_value=FakeResponse([], status_code=500, text="boom")):
            with self.assertRaisesRegex(LocalLLMError, "HTTP 500"):
                client.chat("x")

    def test_empty_response_raises(self) -> None:
        client = OllamaChatClient("http://127.0.0.1:9", MODEL, retry_attempts=0)
        with patch("src.llm.ollama_chat.requests.post", return_value=FakeResponse([_line(done=True, done_reason="length")])):
            with self.assertRaisesRegex(LocalLLMError, "빈 응답"):
                client.chat("x")


class DryRunClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OllamaChatClient("http://127.0.0.1:9", MODEL, dry_run=True)

    def test_status_and_health(self) -> None:
        self.assertTrue(self.client.is_ready())
        status = self.client.status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["health"]["model"], "Gemma4-26B-A4B:Q4_K_M")

    def test_chat_returns_korean_prose_and_streams(self) -> None:
        received: list[str] = []
        text = self.client.chat("다음 장면을 써줘", system="작가", stream_callback=received.append)
        self.assertIn("서윤", text)
        self.assertEqual("".join(received), text)

    def test_json_mode_serves_every_parser(self) -> None:
        payload = json.loads(self.client.chat("작품 카드를 JSON 으로", json_mode=True))
        self.assertEqual(payload["card"]["title"], "유리등의 속삭임")
        self.assertEqual(payload["score"], 8)
        self.assertIn("goal", payload)
        self.assertIn("beats", payload)

    def test_count_tokens_estimates_from_characters(self) -> None:
        self.assertEqual(self.client.count_tokens(text="가" * 150), 100)
        self.assertGreater(self.client.count_tokens(messages=[{"role": "user", "content": "안녕하세요 반갑습니다"}]), 0)

    def test_embed_is_unavailable(self) -> None:
        with self.assertRaises(LocalLLMError):
            self.client.embed(["text"])


class UnreachableServerTests(unittest.TestCase):
    def test_status_reports_connection_failure(self) -> None:
        status = OllamaChatClient("http://127.0.0.1:9", MODEL, retry_attempts=0).status()
        self.assertFalse(status["ready"])
        self.assertIn("연결", status["reason"])

    def test_chat_raises_clear_error(self) -> None:
        client = OllamaChatClient("http://127.0.0.1:9", MODEL, retry_attempts=0, retry_backoff_sec=0)
        with self.assertRaises(LocalLLMError):
            client.chat("hello")


class HealthTests(unittest.TestCase):
    def test_missing_model_reports_the_pull_command(self) -> None:
        tags = FakeResponse([], text=json.dumps({"models": [{"name": "gemma4:e4b"}]}))
        with patch("src.llm.ollama_chat.requests.get", return_value=tags):
            status = OllamaChatClient("http://127.0.0.1:9", MODEL).status()
        self.assertFalse(status["ready"])
        self.assertIn(f"ollama pull {MODEL}", status["reason"])
        self.assertEqual(status["health"]["status"], "missing")

    def test_installed_model_reports_ready_and_vram(self) -> None:
        def fake_get(url: str, timeout: int = 5) -> FakeResponse:
            if url.endswith("/api/tags"):
                return FakeResponse([], text=json.dumps({"models": [{"name": MODEL}]}))
            return FakeResponse([], text=json.dumps({"models": [{"name": MODEL, "size_vram": 5 * 2**30,
                                                                  "context_length": 16384}]}))

        with patch("src.llm.ollama_chat.requests.get", side_effect=fake_get):
            status = OllamaChatClient("http://127.0.0.1:9", MODEL).status()
        self.assertTrue(status["ready"])
        health = status["health"]
        self.assertTrue(health["loaded"])
        self.assertEqual(health["vram_gib"], 5.0)
        self.assertEqual(health["context_length"], 16384)
        self.assertEqual(health["model"], "Gemma4-26B-A4B:Q4_K_M")

    def test_model_matching_ignores_the_latest_tag(self) -> None:
        self.assertTrue(model_installed("embeddinggemma", ["embeddinggemma:latest"]))
        self.assertTrue(model_installed("Gemma4:E4B", ["gemma4:e4b"]))
        self.assertFalse(model_installed("m:q4", ["m:q8"]))

    def test_a_missing_model_name_is_rejected_early(self) -> None:
        with self.assertRaises(ValueError):
            OllamaChatClient("http://127.0.0.1:9", "")


if __name__ == "__main__":
    unittest.main()
