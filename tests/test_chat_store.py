from __future__ import annotations

import tempfile
import unittest

from src.service.chat_store import ChatStore, ChatStoreError
from src.service.consumer_store import ConsumerStore
from src.utils.config import AppConfig


class ChatStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        # 같은 SQLite 파일을 작품 저장소와 공유한다.
        self.consumer = ConsumerStore(self.config)
        self.store = ChatStore(self.config)
        self.user = self.consumer.create_user(username="chat_user", display_name="Chat", password="chat-password")
        self.other = self.consumer.create_user(username="other_user", display_name="Other", password="other-password")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create_append_and_title_from_first_message(self) -> None:
        chat = self.store.create_chat(self.user["id"])
        self.assertEqual(chat["title"], "새 대화")
        self.store.append_message(self.user["id"], chat["id"], "user", "   오늘 저녁 뭐 먹을까?  ")
        self.store.append_message(self.user["id"], chat["id"], "assistant", "김치찌개 어때?")
        messages = self.store.list_messages(self.user["id"], chat["id"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "오늘 저녁 뭐 먹을까?")
        listed = self.store.list_chats(self.user["id"])
        self.assertEqual(listed[0]["title"], "오늘 저녁 뭐 먹을까?")
        self.assertEqual(listed[0]["message_count"], 2)

    def test_ownership_is_enforced(self) -> None:
        chat = self.store.create_chat(self.user["id"])
        self.assertIsNone(self.store.get_chat(self.other["id"], chat["id"]))
        with self.assertRaises(ChatStoreError):
            self.store.append_message(self.other["id"], chat["id"], "user", "침입")
        with self.assertRaises(ChatStoreError):
            self.store.delete_chat(self.other["id"], chat["id"])

    def test_adapter_rename_delete_pair_and_clear(self) -> None:
        chat = self.store.create_chat(self.user["id"], adapter="")
        self.store.set_adapter(self.user["id"], chat["id"], "cute")
        self.assertEqual(self.store.get_chat(self.user["id"], chat["id"])["adapter"], "cute")
        self.store.rename_chat(self.user["id"], chat["id"], "말투 실험")
        self.assertEqual(self.store.get_chat(self.user["id"], chat["id"])["title"], "말투 실험")
        first = self.store.append_message(self.user["id"], chat["id"], "user", "첫 질문")
        self.store.append_message(self.user["id"], chat["id"], "assistant", "첫 답변")
        self.store.append_message(self.user["id"], chat["id"], "user", "둘째 질문")
        self.store.delete_message_pair(self.user["id"], chat["id"], int(first["id"]))
        remaining = self.store.list_messages(self.user["id"], chat["id"])
        self.assertEqual([m["content"] for m in remaining], ["둘째 질문"])
        self.store.clear_messages(self.user["id"], chat["id"])
        self.assertEqual(self.store.list_messages(self.user["id"], chat["id"]), [])
        self.store.delete_chat(self.user["id"], chat["id"])
        self.assertEqual(self.store.list_chats(self.user["id"]), [])

    def test_empty_and_invalid_messages_are_rejected(self) -> None:
        chat = self.store.create_chat(self.user["id"])
        with self.assertRaises(ChatStoreError):
            self.store.append_message(self.user["id"], chat["id"], "user", "   ")
        with self.assertRaises(ChatStoreError):
            self.store.append_message(self.user["id"], chat["id"], "tool", "x")


if __name__ == "__main__":
    unittest.main()
