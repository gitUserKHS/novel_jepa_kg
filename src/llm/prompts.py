from __future__ import annotations

import json


def synthetic_sample_prompt(genre: str, sample_index: int) -> str:
    return f"""
한국어 소설 장면 전환 학습 데이터를 JSON 한 개로 작성하세요. sample #{sample_index}

장르: {genre}

반드시 아래 최상위 필드를 포함하세요:
- world: genre, premise, rules
- characters: name, role, goal, fear, relationship을 가진 인물 목록
- scene_t: summary, emotion, conflict, state, plot_function
- scene_t_plus_1: summary, emotion, conflict, state, plot_function

조건:
- 한국어로만 작성합니다.
- 실제 저작권 프랜차이즈명, 유명 캐릭터명은 쓰지 않습니다.
- scene_t와 scene_t_plus_1은 같은 사건을 반복하지 말고 원인과 결과가 이어져야 합니다.
- JSON 외의 설명, 마크다운 코드펜스는 출력하지 않습니다.
"""


def prose_prompt(
    world: str,
    characters: str,
    previous_scene: str,
    style: str,
    direction: str | None = None,
    examples: list[str] | None = None,
) -> str:
    example_text = "\n".join(f"- {example}" for example in (examples or []))
    return f"""
아래 설정을 바탕으로 다음 장면의 한국어 웹소설 본문을 작성하세요.

[세계관]
{world}

[인물]
{characters}

[이전 장면 요약]
{previous_scene}

[예측된 다음 장면 방향]
{direction or "이전 장면의 갈등을 자연스럽게 진전시킨다."}

[참고 장면 예시]
{example_text or "- 참고 장면 없음"}

[문체 제약]
{style}

출력 조건:
- 본문만 출력합니다.
- 1200자 이내.
- 새 단서, 감정 변화, 다음 선택지를 포함합니다.
"""


def report_header(outputs: dict[str, str]) -> str:
    return json.dumps({key: len(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2)
