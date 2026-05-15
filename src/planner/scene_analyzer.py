from __future__ import annotations

from typing import Any

from src.data.validate_jsonl import parse_json_object
from src.generation.consistency import extract_character_names
from src.llm.ollama_client import OllamaClient
from src.planner.jepa_dataset import _line
from src.utils.config import AppConfig

ANALYZER_FIELDS = [
    "summary",
    "emotion",
    "conflict",
    "state",
    "plot_function",
    "active_characters",
    "unresolved_clues",
    "next_pressure",
]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _fallback_emotion(text: str) -> str:
    if any(token in text for token in ["공포", "두려", "떨", "위험", "불안"]):
        return "불안"
    if any(token in text for token in ["분노", "화가", "배신", "거짓"]):
        return "분노"
    if any(token in text for token in ["결심", "선택", "따라", "향해"]):
        return "결심"
    if any(token in text for token in ["혼란", "낯선", "이상", "모순"]):
        return "혼란"
    return "긴장"


def _fallback_plot_function(text: str) -> str:
    if any(token in text for token in ["발견", "찾", "로그", "기록", "단서"]):
        return "단서 발견"
    if any(token in text for token in ["선택", "결정", "해야", "갈림"]):
        return "선택 압박"
    if any(token in text for token in ["습격", "위험", "추적", "경보", "폭주"]):
        return "위기 고조"
    if any(token in text for token in ["배신", "거짓", "숨긴", "의심"]):
        return "관계 균열"
    return "상태 전환"


def _fallback_clues(text: str) -> list[str]:
    clues = []
    for pattern in ["기록", "로그", "좌표", "이름", "메시지", "장부", "유물", "녹취", "지도", "문서", "사진"]:
        if pattern in text:
            clues.append(pattern)
    return clues[:5] or ["확인되지 않은 단서"]


def _fallback_state(text: str) -> list[str]:
    state = [_fallback_emotion(text), _fallback_plot_function(text)]
    for clue in _fallback_clues(text)[:3]:
        state.append(f"단서:{clue}")
    return state


def fallback_analyze_current_scene(
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None = None,
) -> dict[str, Any]:
    text = " ".join(previous_scene.strip().split())
    preset = scene_preset or {}
    names = [name for name in extract_character_names(characters) if name and name in previous_scene]
    if not names:
        names = extract_character_names(characters)[:3]
    summary = text[:280] if text else "이전 장면 정보 없음"
    plot_function = preset.get("plot_function") or _fallback_plot_function(text)
    conflict = preset.get("conflict") or "이전 장면의 단서를 따라가면 더 큰 위험 또는 선택 압박이 발생한다."
    return {
        "summary": summary,
        "emotion": _fallback_emotion(text),
        "conflict": conflict,
        "state": _fallback_state(text),
        "plot_function": plot_function,
        "active_characters": names,
        "unresolved_clues": _fallback_clues(text),
        "next_pressure": preset.get("next_hook") or "다음 장면에서 단서의 대가와 선택 압박이 드러난다.",
        "source": "fallback",
    }


def _normalize_analysis(payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: payload.get(key, fallback.get(key)) for key in ANALYZER_FIELDS}
    normalized["state"] = _as_list(normalized.get("state")) or fallback["state"]
    normalized["active_characters"] = _as_list(normalized.get("active_characters")) or fallback["active_characters"]
    normalized["unresolved_clues"] = _as_list(normalized.get("unresolved_clues")) or fallback["unresolved_clues"]
    for key in ["summary", "emotion", "conflict", "plot_function", "next_pressure"]:
        value = normalized.get(key)
        normalized[key] = str(value).strip() if value else str(fallback[key])
    normalized["source"] = payload.get("source", "ollama")
    return normalized


def analyze_current_scene(
    config: AppConfig,
    client: OllamaClient,
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None = None,
) -> dict[str, Any]:
    fallback = fallback_analyze_current_scene(world, characters, previous_scene, scene_preset)
    if client.dry_run:
        return fallback

    preset_text = ""
    if scene_preset:
        preset_text = "\n".join(f"- {key}: {value}" for key, value in scene_preset.items() if value)
    prompt = f"""
다음 한국어 장편 소설의 현재 장면을 JEPA-inspired planner 입력용 JSON으로만 구조화하세요.

[세계관]
{world}

[인물표]
{characters}

[현재 장면]
{previous_scene}

[선택된 장면 프리셋]
{preset_text or "(없음)"}

출력 JSON 필드:
- summary: 현재 장면 한 문장 요약
- emotion: 현재 장면의 핵심 감정 상태
- conflict: 현재 장면의 중심 갈등
- state: 현재 장면 상태 태그 문자열 배열
- plot_function: 장면 기능
- active_characters: 현재 장면에서 실제로 활성화된 인물명 배열
- unresolved_clues: 아직 해결되지 않은 단서 배열
- next_pressure: 다음 장면으로 이어지는 압박 또는 훅

조건:
- JSON 객체 하나만 출력합니다.
- 이전 장면에 없는 다음 장면 내용을 만들지 않습니다.
- 입력 인물표에 없는 새 고유명사 인물명을 만들지 않습니다.
"""
    try:
        text = client.chat(
            prompt,
            system="당신은 소설 장면을 구조화하는 분석기입니다. JSON만 출력합니다.",
            temperature=0.2,
            max_tokens=700,
            json_mode=True,
        )
        payload = parse_json_object(text)
    except Exception:
        return fallback
    return _normalize_analysis(payload, fallback)


def build_analyzed_generation_context(
    world: str,
    characters: str,
    previous_scene: str,
    analysis: dict[str, Any],
    scene_preset: dict[str, str] | None = None,
) -> str:
    preset = scene_preset or {}
    return "\n".join(
        [
            "[Context Encoder Input]",
            "[World]",
            _line("세계 설정", world),
            "",
            "[Characters]",
            _line("인물표", characters),
            "",
            "[Current Scene Analysis]",
            _line("요약", analysis.get("summary") or previous_scene),
            _line("감정", analysis.get("emotion")),
            _line("갈등", analysis.get("conflict")),
            _line("상태", analysis.get("state")),
            _line("장면 기능", analysis.get("plot_function")),
            _line("활성 인물", analysis.get("active_characters")),
            _line("미해결 단서", analysis.get("unresolved_clues")),
            _line("다음 압박", analysis.get("next_pressure")),
            "",
            "[Scene Preset Metadata]",
            _line("프리셋", preset.get("label")),
            _line("프리셋 기능", preset.get("plot_function")),
            _line("프리셋 모티프", preset.get("motif")),
            _line("관계 긴장", preset.get("relationship")),
        ]
    )
