"""작품 설정 → 생성 프롬프트에 들어가는 설정 시트 문자열.

워커(장 생성)와 웹 앱(원고 수정·AI 퇴고·메모리 재추출)이 같은 시트를 써야 모델이 같은 설정을 본다.
"""

from __future__ import annotations

from typing import Any


def character_sheet(story: dict[str, Any]) -> str:
    """주인공 + 주요 인물. '이름: 소개' 줄 목록."""
    explicit = str(story.get("characters", "") or "").strip()
    protagonist = str(story.get("protagonist", "") or "").strip()
    if explicit:
        return f"{protagonist}\n{explicit}" if protagonist and protagonist not in explicit else explicit
    if ":" in protagonist or "：" in protagonist:
        return protagonist
    return f"{protagonist}: 작품의 주인공"


def world_sheet(story: dict[str, Any]) -> str:
    """장르·소재·세계관. 이야기 지도와 서사 기능 순환의 씨앗(story_seed)에도 쓰이므로 설정만 담는다."""
    return "\n".join([
        f"장르: {story.get('genre', '')}",
        f"핵심 소재: {story.get('premise', '')}",
        f"세계관: {story.get('world', '')}",
    ])


def style_guide(story: dict[str, Any]) -> str:
    """사용자가 직접 적은 집필 지침(문체·시점·금기). 프롬프트에서는 설정이 아니라 규칙 블록으로 들어간다."""
    return str(story.get("style_guide", "") or "").strip()
