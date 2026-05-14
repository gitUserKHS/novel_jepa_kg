from __future__ import annotations

import json
from typing import Any


DIVERSITY_PLANS: list[dict[str, str]] = [
    {
        "subgenre": "SF 미스터리",
        "plot_function": "단서 발견",
        "emotion_arc": "불안에서 결심으로",
        "conflict": "위험한 단서를 해석할지 선택한다",
        "motif": "기억 잔향, 폐쇄 구역, 손상된 기록",
        "relationship": "협력자에 대한 불신",
    },
    {
        "subgenre": "궁중 판타지",
        "plot_function": "권력 역전",
        "emotion_arc": "굴욕에서 냉정한 반격으로",
        "conflict": "거짓 충성 맹세와 진짜 목표가 충돌한다",
        "motif": "봉인된 칙서, 금지된 예언, 연회장",
        "relationship": "스승과 제자의 숨은 배신",
    },
    {
        "subgenre": "현대 오컬트",
        "plot_function": "금기 위반",
        "emotion_arc": "호기심에서 공포로",
        "conflict": "의식을 멈추면 친구를 잃고 계속하면 재앙이 열린다",
        "motif": "낡은 아파트, 거울, 이름 없는 의식",
        "relationship": "오랜 친구 사이의 죄책감",
    },
    {
        "subgenre": "로맨스 스릴러",
        "plot_function": "관계 균열",
        "emotion_arc": "설렘에서 의심으로",
        "conflict": "사랑하는 사람이 사건의 핵심 증거를 숨긴다",
        "motif": "비 오는 플랫폼, 사라진 메시지, 낯선 향수",
        "relationship": "연인 사이의 신뢰 붕괴",
    },
    {
        "subgenre": "무협 정치극",
        "plot_function": "동맹 제안",
        "emotion_arc": "분노에서 전략적 인내로",
        "conflict": "원수의 제안을 받아야만 더 큰 음모에 접근한다",
        "motif": "비무대, 독문서, 무너진 문파",
        "relationship": "원수와 임시 동맹",
    },
    {
        "subgenre": "디스토피아 성장물",
        "plot_function": "정체성 각성",
        "emotion_arc": "순응에서 저항으로",
        "conflict": "안전한 거짓 삶과 위험한 진실 중 하나를 고른다",
        "motif": "배급표, 감시 드론, 금지된 노래",
        "relationship": "가족을 지키려는 침묵",
    },
    {
        "subgenre": "해양 모험",
        "plot_function": "목표 확장",
        "emotion_arc": "경계에서 경이로",
        "conflict": "난파선을 구하면 추격자가 따라오고 버리면 단서를 잃는다",
        "motif": "검은 등대, 심해 지도, 폭풍 전야",
        "relationship": "선장과 항해사의 책임 충돌",
    },
    {
        "subgenre": "법정 미스터리",
        "plot_function": "증언 반전",
        "emotion_arc": "확신에서 혼란으로",
        "conflict": "승소를 위해 의뢰인의 거짓말을 폭로해야 한다",
        "motif": "녹취록, 휴정 직전, 사라진 목격자",
        "relationship": "변호사와 의뢰인의 도덕적 대립",
    },
    {
        "subgenre": "학원 이능 배틀",
        "plot_function": "능력 대가 공개",
        "emotion_arc": "자만에서 두려움으로",
        "conflict": "힘을 쓰면 기억이 사라지지만 쓰지 않으면 동료가 다친다",
        "motif": "훈련장, 금 간 교표, 응급 방송",
        "relationship": "라이벌과의 미묘한 신뢰",
    },
    {
        "subgenre": "가족 드라마 미스터리",
        "plot_function": "비밀 상속",
        "emotion_arc": "원망에서 연민으로",
        "conflict": "상속 문서가 가족의 오래된 죄를 드러낸다",
        "motif": "빈집, 오래된 장부, 닫힌 다락",
        "relationship": "남매 사이의 오래된 오해",
    },
    {
        "subgenre": "사이버펑크 누아르",
        "plot_function": "추적 전환",
        "emotion_arc": "냉소에서 집착으로",
        "conflict": "의뢰인이 피해자가 아니라 설계자일 수 있다",
        "motif": "네온 골목, 불법 백업, 깨진 의안",
        "relationship": "탐정과 정보상의 거래",
    },
    {
        "subgenre": "역사 대체물",
        "plot_function": "역사 분기",
        "emotion_arc": "충성에서 회의로",
        "conflict": "왕명을 따르면 전쟁이 나고 어기면 반역자가 된다",
        "motif": "밀서, 새벽 성문, 바뀐 연호",
        "relationship": "군주와 신하의 믿음 시험",
    },
]


def diversity_plan(sample_index: int, bucket_count: int | None = None) -> dict[str, str]:
    plans = DIVERSITY_PLANS[: bucket_count or len(DIVERSITY_PLANS)]
    return plans[(sample_index - 1) % len(plans)]


def synthetic_sample_prompt(genre: str, sample_index: int, plan: dict[str, str] | None = None) -> str:
    selected = plan or diversity_plan(sample_index)
    return f"""
한국어 소설 장면 전환 학습 데이터를 JSON 한 개로 작성하세요. sample #{sample_index}

입력 장르: {genre}
이번 샘플의 다양화 플랜:
- 세부 장르: {selected["subgenre"]}
- 장면 기능: {selected["plot_function"]}
- 감정 변화: {selected["emotion_arc"]}
- 핵심 갈등: {selected["conflict"]}
- 모티프: {selected["motif"]}
- 관계 긴장: {selected["relationship"]}

반드시 아래 최상위 필드를 포함하세요:
- world: genre, premise, rules
- characters: name, role, goal, fear, relationship을 가진 인물 목록
- scene_t: summary, emotion, conflict, state, plot_function
- scene_t_plus_1: summary, emotion, conflict, state, plot_function

조건:
- 한국어로만 작성합니다.
- 실제 저작권 프랜차이즈명, 유명 캐릭터명은 쓰지 않습니다.
- scene_t와 scene_t_plus_1은 같은 사건을 반복하지 말고 원인과 결과가 이어져야 합니다.
- scene_t_plus_1에는 새 단서, 선택 압박, 관계 변화 중 최소 두 가지를 포함합니다.
- 매 샘플마다 장소, 목표, 갈등, 감정선을 다르게 만듭니다.
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
- 이전 장면을 반복 요약하지 말고, 사건의 상태를 한 단계 바꿉니다.
"""


def report_header(outputs: dict[str, str]) -> str:
    return json.dumps({key: len(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2)


def compact_plan_text(plan: dict[str, Any]) -> str:
    return " / ".join(str(value) for value in plan.values())
