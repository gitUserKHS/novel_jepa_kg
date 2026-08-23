from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from src.utils.timefmt import DATE_GROUPS, date_group, parse_local, relative_time


# 로컬 시간대 기준 (함수가 로컬 날짜로 묶으므로 시험도 로컬 시각으로 고정한다)
NOW = datetime(2026, 8, 23, 15, 30).astimezone()


def _ago(**kwargs: int) -> str:
    return (NOW - timedelta(**kwargs)).isoformat(timespec="seconds")


class TimeFormatTests(unittest.TestCase):
    def test_relative_time_buckets(self) -> None:
        cases = {
            _ago(seconds=5): "방금 전",
            _ago(minutes=3): "3분 전",
            _ago(hours=2): "2시간 전",
            _ago(days=1): "어제",
            _ago(days=4): "4일 전",
            _ago(days=20): "8월 3일",
            _ago(days=400): "2025년 7월 19일",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(relative_time(value, now=NOW), expected)

    def test_date_groups(self) -> None:
        cases = {
            _ago(minutes=1): "오늘",
            _ago(days=1): "어제",
            _ago(days=6): "지난 7일",
            _ago(days=7): "지난 30일",
            _ago(days=29): "지난 30일",
            _ago(days=30): "이전",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(date_group(value, now=NOW), expected)
        self.assertEqual(DATE_GROUPS[0], "오늘")

    def test_naive_values_are_read_as_utc(self) -> None:
        from datetime import UTC

        naive = (NOW - timedelta(minutes=10)).astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        self.assertEqual(relative_time(naive, now=NOW), "10분 전")
        self.assertIsNotNone(parse_local(naive))

    def test_garbage_is_harmless(self) -> None:
        self.assertEqual(relative_time("", now=NOW), "")
        self.assertEqual(relative_time("not-a-date", now=NOW), "")
        self.assertIsNone(parse_local(None))
        self.assertEqual(date_group("not-a-date", now=NOW), "이전")


if __name__ == "__main__":
    unittest.main()
