from __future__ import annotations

import tempfile
import unittest

from src.service.consumer_store import ConsumerStore
from src.service.template_store import StoryTemplateStore, TemplateStoreError, template_from_story
from src.utils.config import AppConfig


class StoryTemplateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = AppConfig(output_root=self.temporary.name)
        self.consumer = ConsumerStore(self.config)
        self.store = StoryTemplateStore(self.config)
        self.user = self.consumer.create_user(username="tpl_user", display_name="T", password="tpl-password")
        self.other = self.consumer.create_user(username="tpl_other", display_name="O", password="other-password")
        self.uid = str(self.user["id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_list_get_delete(self) -> None:
        saved = self.store.save_template(
            self.uid, name="  항구 미스터리  ", genre="미스터리", world="비 오는 항구 도시", protagonist="등대지기 딸",
            style_guide="담백한 문체", target_chars=30000,
        )
        self.assertEqual(saved["name"], "항구 미스터리")
        self.assertEqual(saved["world"], "비 오는 항구 도시")
        self.assertEqual(saved["target_chars"], 30000)
        listed = self.store.list_templates(self.uid)
        self.assertEqual([t["name"] for t in listed], ["항구 미스터리"])
        self.assertEqual(self.store.get_template(self.uid, saved["id"])["style_guide"], "담백한 문체")
        self.assertIsNone(self.store.get_template(str(self.other["id"]), saved["id"]))
        self.assertEqual(self.store.list_templates(str(self.other["id"])), [])
        with self.assertRaises(TemplateStoreError):
            self.store.delete_template(str(self.other["id"]), saved["id"])
        self.store.delete_template(self.uid, saved["id"])
        self.assertEqual(self.store.list_templates(self.uid), [])

    def test_same_name_overwrites_and_moves_to_the_top(self) -> None:
        first = self.store.save_template(self.uid, name="A", genre="SF")
        self.store.save_template(self.uid, name="B", genre="판타지")
        second = self.store.save_template(self.uid, name="A", genre="SF 미스터리", world="근미래 서울")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["genre"], "SF 미스터리")
        self.assertEqual([t["name"] for t in self.store.list_templates(self.uid)], ["A", "B"])

    def test_validation(self) -> None:
        with self.assertRaises(TemplateStoreError):
            self.store.save_template(self.uid, name="   ", genre="SF")
        with self.assertRaises(TemplateStoreError):
            self.store.save_template(self.uid, name="빈 템플릿", title="제목만")
        with self.assertRaises(TemplateStoreError):
            self.store.save_template(self.uid, name="x", genre="SF", bogus="y")  # type: ignore[arg-type]
        long = self.store.save_template(self.uid, name="길이", world="w" * 5000)
        self.assertEqual(len(long["world"]), 4000)

    def test_template_from_story_keeps_only_template_fields(self) -> None:
        story = {"id": "abc", "title": "t", "genre": "g", "premise": "p", "world": "w", "protagonist": "x",
                 "characters": "", "style_guide": "s", "current_chars": 10}
        self.assertEqual(
            template_from_story(story),
            {"title": "t", "genre": "g", "premise": "p", "world": "w", "protagonist": "x", "characters": "", "style_guide": "s"},
        )


if __name__ == "__main__":
    unittest.main()
