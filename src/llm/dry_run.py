"""모델 없이 돌리는 고정 응답 (테스트·CI, `NOVEL_LLM_DRY_RUN=1`).

Ollama 클라이언트와 레거시 로컬 서버 클라이언트가 같은 응답을 쓴다. JSON 응답은 기획 카드·이야기 지도·
장 설계·개연성 검토·요약 메모리 파서가 각자 필요한 키만 읽는 하나의 묶음이다 — 파서는 모르는 키를 무시한다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

DRY_JSON: dict[str, object] = {
    # 기획 대화 (consumer_app._plan_turn)
    "reply": "설정을 정리했어. 이대로 시작할까?",
    "ready": True,
    "card": {
        "title": "유리등의 속삭임",
        "genre": "SF 미스터리",
        "premise": "비가 멈춘 도시에서 사라진 기억을 추적한다",
        "world": "기억이 거래되는 근미래 서울",
        "protagonist": "서윤: 기록 복원가",
        "characters": "민재: 핵심 단서를 숨긴 연구원",
        "target_chars": 30000,
    },
    # 이야기 지도 (story_outline.create_story_outline)
    "premise": "주인공의 선택이 도시의 균형을 바꾼다",
    "ending_intent": "대가를 치르고 진실을 택한다",
    # create_story_outline 은 beat 가 4개 미만이면 모델 응답을 버리고 기본 지도로 물러난다 — 4개를 둔다.
    "beats": [
        {"beat_id": 1, "phase": "1막", "purpose": "균형을 깨는 사건", "required_change": "주인공이 문제를 외면할 수 없게 된다"},
        {"beat_id": 2, "phase": "2막", "purpose": "선택과 대가", "required_change": "동맹의 조건이 달라진다"},
        {"beat_id": 3, "phase": "2막", "purpose": "반격과 손실", "required_change": "안전한 선택지가 하나 사라진다"},
        {"beat_id": 4, "phase": "3막", "purpose": "수렴과 결말", "required_change": "중심 질문에 답한다"},
    ],
    # 장 설계 (longform._plan_section)
    "goal": "서윤이 심층 구역 좌표의 출처를 확인한다",
    "obstacle": "민재가 좌표의 출처를 숨긴다",
    "turn": "기록 장치가 동생의 목소리를 재생한다",
    "outcome": "서윤이 민재를 의심하며 심층 구역으로 향할 결심을 한다",
    "uses": ["심층 구역 좌표"],
    "avoid": ["첫 단서 발견 장면의 반복"],
    # 개연성 검토 (plausibility.review_section) — 대사 화자 점검은 dialogue 를 읽는다
    "non_speakers": [],
    "dialogue": [],
    "score": 8,
    "contradictions": [],
    "unmotivated": [],
    "continuity_ok": True,
    "beat_progress": True,
    "verdict": "설정과 앞선 사건에 어긋나는 점이 없다.",
    # 요약 메모리 (longform._extract_memory)
    "section_index": 1,
    "summary": "서윤이 첫 단서를 얻는다.",
    "characters_list": ["서윤", "민재"],
    "facts": ["기록 장치가 동생의 목소리를 재생했다"],
    "open_clues": ["심층 구역 좌표"],
    "resolved_clues": [],
    "locations": ["폐쇄 연구동"],
    "state_changes": ["서윤이 좌표를 확보했다"],
    "keywords": ["기록", "좌표", "동생"],
}

DRY_PROSE = (
    "### 젖은 골목의 신호\n\n"
    "서윤은 차가운 형광등 아래에서 숨을 골랐다. 기록 장치가 토해낸 잔향은 동생의 목소리였지만, "
    "그 안에는 도망치는 사람의 공포보다 무언가를 선택한 사람의 단단함이 남아 있었다.\n\n"
    "민재는 심층 구역 좌표를 보는 순간 얼굴빛을 잃었다. 서윤은 그 침묵이 대답이라는 걸 알았다. "
    "이제 그녀가 찾아야 할 것은 동생의 행방만이 아니었다. 왜 모두가 그 선택을 숨기려 했는지, "
    "그리고 자신이 잃어버린 첫 번째 기억이 무엇인지 확인해야 했다.\n\n"
    "비는 그치지 않았다. 서윤은 젖은 골목 끝에서 다시 한 번 장치를 켰다."
)


def dry_response(prompt: str, json_mode: bool) -> str:
    """프롬프트가 JSON 을 요구하면 고정 JSON, 아니면 고정 산문."""
    if json_mode or "JSON" in prompt:
        return json.dumps(DRY_JSON, ensure_ascii=False)
    return DRY_PROSE


def dry_stream(prompt: str, json_mode: bool, chunk_chars: int = 24) -> Iterator[str]:
    text = dry_response(prompt, json_mode)
    if json_mode or "JSON" in prompt:
        yield text
        return
    for start in range(0, len(text), chunk_chars):
        yield text[start : start + chunk_chars]
