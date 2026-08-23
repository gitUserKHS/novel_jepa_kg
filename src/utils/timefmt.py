"""저장소의 ISO 시각(UTC)을 사용자에게 보여줄 한국어 상대 시각·날짜 묶음으로 바꾼다.

순수 함수만 둔다. `now` 를 넘기면 시험에서 결과를 고정할 수 있다.
"""

from __future__ import annotations

from datetime import UTC, datetime

DATE_GROUPS = ("오늘", "어제", "지난 7일", "지난 30일", "이전")


def local_now() -> datetime:
    return datetime.now().astimezone()


def parse_local(value: str | None) -> datetime | None:
    """ISO 문자열을 로컬 시간대의 aware datetime 으로. 시간대가 없으면 UTC 로 본다."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone()


def relative_time(value: str | None, *, now: datetime | None = None) -> str:
    """'방금 전', '3분 전', '2시간 전', '어제', '4일 전', '8월 12일', '2025년 12월 3일'."""
    moment = parse_local(value)
    if moment is None:
        return ""
    current = (now or local_now()).astimezone()
    seconds = (current - moment).total_seconds()
    if seconds < 60:
        return "방금 전"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    days = (current.date() - moment.date()).days
    if days <= 0:
        return f"{int(seconds // 3600)}시간 전"
    if days == 1:
        return "어제"
    if days < 7:
        return f"{days}일 전"
    if moment.year == current.year:
        return f"{moment.month}월 {moment.day}일"
    return f"{moment.year}년 {moment.month}월 {moment.day}일"


def date_group(value: str | None, *, now: datetime | None = None) -> str:
    """목록을 묶는 날짜 구간 이름. 시각을 못 읽으면 '이전'."""
    moment = parse_local(value)
    if moment is None:
        return DATE_GROUPS[-1]
    current = (now or local_now()).astimezone()
    days = (current.date() - moment.date()).days
    if days <= 0:
        return "오늘"
    if days == 1:
        return "어제"
    if days < 7:
        return "지난 7일"
    if days < 30:
        return "지난 30일"
    return "이전"
