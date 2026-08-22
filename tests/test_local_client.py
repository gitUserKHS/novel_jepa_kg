from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.llm.local_client import LocalLLMClient, LocalLLMError, _iter_sse


def _sse_response(lines: list[str]) -> SimpleNamespace:
    return SimpleNamespace(iter_lines=lambda decode_unicode=True: iter(lines))


class SSEParserTests(unittest.TestCase):
    def test_deltas_and_done(self) -> None:
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "안녕"}, "finish_reason": None}]}),
            "",
            "data: " + json.dumps({"choices": [{"delta": {"content": "하세요"}, "finish_reason": None}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}),
            "data: [DONE]",
            "data: " + json.dumps({"choices": [{"delta": {"content": "무시"}, "finish_reason": None}]}),
        ]
        pieces = list(_iter_sse(_sse_response(lines)))
        self.assertEqual([piece for piece, _final in pieces], ["안녕", "하세요"])

    def test_stop_string_final_text_is_surfaced(self) -> None:
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "본문 ###"}, "finish_reason": None}]}),
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "text": "본문 "}),
            "data: [DONE]",
        ]
        pieces = list(_iter_sse(_sse_response(lines)))
        self.assertEqual(pieces[-1], ("", "본문 "))

    def test_server_error_raises(self) -> None:
        lines = ["data: " + json.dumps({"error": "prompt too long"})]
        with self.assertRaisesRegex(LocalLLMError, "prompt too long"):
            list(_iter_sse(_sse_response(lines)))


class DryRunClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LocalLLMClient("http://127.0.0.1:9", dry_run=True)

    def test_status_and_health(self) -> None:
        self.assertTrue(self.client.is_ready())
        self.assertTrue(self.client.status()["ready"])

    def test_chat_returns_korean_prose_and_streams(self) -> None:
        received: list[str] = []
        text = self.client.chat("다음 장면을 써줘", system="작가", stream_callback=received.append)
        self.assertIn("서윤", text)
        self.assertEqual("".join(received), text)

    def test_json_mode_is_parseable(self) -> None:
        payload = json.loads(self.client.chat("작품 카드를 JSON 으로", json_mode=True))
        self.assertIn("card", payload)
        self.assertEqual(payload["card"]["title"], "유리등의 속삭임")

    def test_count_tokens(self) -> None:
        self.assertGreater(self.client.count_tokens(text="안녕하세요 반갑습니다"), 0)

    def test_embed_is_unavailable(self) -> None:
        with self.assertRaises(LocalLLMError):
            self.client.embed(["text"])


class UnreachableServerTests(unittest.TestCase):
    def test_status_reports_connection_failure(self) -> None:
        client = LocalLLMClient("http://127.0.0.1:9", retry_attempts=0)
        status = client.status()
        self.assertFalse(status["ready"])
        self.assertIn("연결", status["reason"])

    def test_chat_raises_clear_error(self) -> None:
        client = LocalLLMClient("http://127.0.0.1:9", retry_attempts=0, retry_backoff_sec=0)
        with self.assertRaises(LocalLLMError):
            client.chat("hello")


if __name__ == "__main__":
    unittest.main()
