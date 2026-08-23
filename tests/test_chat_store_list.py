from __future__ import annotations

import tempfile
import unittest

from src.service.chat_store import ChatStore, ChatStoreError, export_chat_markdown
from src.service.consumer_store import ConsumerStore
from src.utils.config import AppConfig


class ChatListManagementTests(unittest.TestCase):
    """사이드바 대화 목록이 기대는 저장소 동작: 검색, 빈 대화 재사용, 전체 삭제, 단일 메시지 삭제, 내보내기."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.consumer = ConsumerStore(self.config)
        self.store = ChatStore(self.config)
        self.user = self.consumer.create_user(username="list_user", display_name="List", password="list-password")
        self.other = self.consumer.create_user(username="list_other", display_name="Other", password="other-password")
        self.uid = str(self.user["id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _chat_with(self, *turns: str) -> dict:
        chat = self.store.create_chat(self.uid)
        for index, text in enumerate(turns):
            self.store.append_message(self.uid, chat["id"], "user" if index % 2 == 0 else "assistant", text)
        return chat

    def test_search_matches_title_or_message_body(self) -> None:
        recipe = self._chat_with("김치찌개 레시피 알려줘", "돼지고기를 먼저 볶아.")
        travel = self._chat_with("제주도 여행 일정 짜줘", "첫날은 동쪽 해안.")
        self.store.rename_chat(self.uid, travel["id"], "여름 휴가 계획")

        by_title = self.store.list_chats(self.uid, query="휴가")
        self.assertEqual([c["id"] for c in by_title], [travel["id"]])

        by_body = self.store.list_chats(self.uid, query="돼지고기")
        self.assertEqual([c["id"] for c in by_body], [recipe["id"]])

        self.assertEqual(len(self.store.list_chats(self.uid, query="   ")), 2)
        self.assertEqual(self.store.list_chats(self.uid, query="없는말"), [])

    def test_search_treats_like_wildcards_literally(self) -> None:
        self._chat_with("100% 확실해?", "아니.")
        self._chat_with("snake_case 가 뭐야", "밑줄로 잇는 표기.")
        self.assertEqual(len(self.store.list_chats(self.uid, query="%")), 1)
        self.assertEqual(len(self.store.list_chats(self.uid, query="_case")), 1)
        self.assertEqual(len(self.store.list_chats(self.uid, query="\\")), 0)

    def test_search_is_scoped_to_the_owner(self) -> None:
        self._chat_with("내 비밀 계획", "응.")
        self.assertEqual(self.store.list_chats(str(self.other["id"]), query="비밀"), [])

    def test_find_empty_chat_prefers_the_most_recent_one(self) -> None:
        self.assertIsNone(self.store.find_empty_chat(self.uid))
        self._chat_with("안녕", "반가워")
        older = self.store.create_chat(self.uid)
        newer = self.store.create_chat(self.uid)
        self.store.touch_chat(self.uid, older["id"])
        found = self.store.find_empty_chat(self.uid)
        self.assertIsNotNone(found)
        self.assertIn(found["id"], {older["id"], newer["id"]})
        listed = [c["id"] for c in self.store.list_chats(self.uid)]
        self.assertEqual(listed[0], older["id"], "touch_chat should move the chat to the top")
        self.assertIsNone(self.store.find_empty_chat(str(self.other["id"])))

    def test_touch_chat_requires_ownership(self) -> None:
        chat = self.store.create_chat(self.uid)
        with self.assertRaises(ChatStoreError):
            self.store.touch_chat(str(self.other["id"]), chat["id"])

    def test_delete_all_chats_only_touches_the_owner(self) -> None:
        self._chat_with("하나", "둘")
        self._chat_with("셋")
        foreign = self.store.create_chat(str(self.other["id"]))
        removed = self.store.delete_all_chats(self.uid)
        self.assertEqual(removed, 2)
        self.assertEqual(self.store.list_chats(self.uid), [])
        self.assertIsNotNone(self.store.get_chat(str(self.other["id"]), foreign["id"]))
        self.assertEqual(self.store.delete_all_chats(self.uid), 0)

    def test_delete_messages_from_drops_that_turn_and_everything_after(self) -> None:
        chat = self._chat_with("첫 질문", "첫 답변", "둘째 질문", "둘째 답변")
        messages = self.store.list_messages(self.uid, chat["id"])
        # 마지막 답변만: '다시 생성' 의 준비 단계
        self.assertEqual(self.store.delete_messages_from(self.uid, chat["id"], int(messages[-1]["id"])), 1)
        self.assertEqual([m["content"] for m in self.store.list_messages(self.uid, chat["id"])],
                         ["첫 질문", "첫 답변", "둘째 질문"])
        # 첫 답변부터: '고쳐서 다시 보내기' 가 뒤를 버리는 방식
        self.assertEqual(self.store.delete_messages_from(self.uid, chat["id"], int(messages[1]["id"])), 2)
        self.assertEqual([m["content"] for m in self.store.list_messages(self.uid, chat["id"])], ["첫 질문"])
        with self.assertRaises(ChatStoreError):
            self.store.delete_messages_from(str(self.other["id"]), chat["id"], int(messages[0]["id"]))
        # 다른 대화의 메시지 id 를 넘겨도 이 대화 밖은 건드리지 않는다.
        other_chat = self._chat_with("남의 질문", "남의 답변")
        foreign_first = int(self.store.list_messages(self.uid, other_chat["id"])[0]["id"])
        self.store.delete_messages_from(self.uid, chat["id"], foreign_first)
        self.assertEqual(len(self.store.list_messages(self.uid, other_chat["id"])), 2)

    def test_export_markdown_contains_title_and_turns(self) -> None:
        chat = self._chat_with("오늘 날씨 어때?", "맑고 덥다.")
        text = export_chat_markdown(
            self.store.get_chat(self.uid, chat["id"]), self.store.list_messages(self.uid, chat["id"])
        )
        self.assertTrue(text.startswith("# 오늘 날씨 어때?\n"))
        self.assertIn("**나**\n\n오늘 날씨 어때?", text)
        self.assertIn("**AI**\n\n맑고 덥다.", text)
        self.assertIn("메시지 2개", text)
        self.assertTrue(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
