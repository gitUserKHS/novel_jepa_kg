from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.ollama_client import OllamaClient
    from src.utils.config import AppConfig


NAME_WITH_COLON_RE = re.compile(r"(?:^|[\n.;])\s*([가-힣A-Za-z][가-힣A-Za-z0-9_ ]{0,20}?)\s*[:：]")
HONORIFIC_RE = re.compile(r"([가-힣]{2,4})\s*(?:씨|님|군|양)\b")
QUOTED_NAME_RE = re.compile(r"[\"'“”‘’]([가-힣]{2,4})[\"'“”‘’]")
TOKEN_RE = re.compile(r"[가-힣]{2,5}")
PARTICLES = ("에게서", "에게", "으로", "에서", "까지", "부터", "처럼", "보다", "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "만", "에")


@dataclass(frozen=True)
class ConsistencyCheck:
    known_names: list[str]
    issues: list[str]

    @property
    def score(self) -> float:
        if not self.known_names:
            return 1.0
        return max(0.0, 1.0 - min(1.0, len(self.issues) * 0.25))


def extract_character_names(characters: str) -> list[str]:
    names: list[str] = []
    for match in NAME_WITH_COLON_RE.findall(characters):
        candidate = " ".join(match.strip().split())
        if 1 <= len(candidate) <= 12 and candidate not in names:
            names.append(candidate)
    return names


def allowed_name_instruction(characters: str) -> str:
    names = extract_character_names(characters)
    if not names:
        return "입력 인물표에 없는 새 고유명사 인물명을 만들지 마세요."
    joined = ", ".join(names)
    return (
        f"허용 인물명: {joined}. "
        "이 목록에 없는 새 인물명, 이름 변형, 철자 변형을 만들지 마세요. "
        "이름이 확정되지 않은 가족/연구원/목격자는 고유명사 대신 역할명으로 쓰세요."
    )


def build_beat_card(
    mode: str,
    direction: str | None,
    examples: list[str],
    characters: str,
    context_limit: int,
) -> str:
    names = extract_character_names(characters)
    active_names = ", ".join(names) if names else "입력 인물표의 인물"
    evidence = examples[: max(0, context_limit)]
    evidence_lines = "\n".join(f"  - {item}" for item in evidence) if evidence else "  - 참고 예시 없음"
    return "\n".join(
        [
            f"- 생성 모드: {mode}",
            f"- 중심 인물: {active_names}",
            f"- 다음 의미 방향: {direction or '이전 장면의 갈등을 한 단계 진전'}",
            "- 장면 목표: 한 장면 안에서 하나의 핵심 사건만 전진시킨다.",
            "- 필수 beat: 목표 확인 -> 위험/저항 발생 -> 새 단서 또는 선택 압박 -> 다음 훅.",
            "- RAG 근거는 사실 복사가 아니라 전환 논리만 참고한다.",
            "- 이번 장면에서 새 장치/비밀/위협을 과하게 늘리지 않는다.",
            "- 참고 전환 근거:",
            evidence_lines,
        ]
    )


def check_name_consistency(text: str, characters: str) -> ConsistencyCheck:
    known_names = extract_character_names(characters)
    if not known_names:
        return ConsistencyCheck(known_names=[], issues=[])

    known_set = set(known_names)
    found: list[str] = []
    found.extend(HONORIFIC_RE.findall(text))
    found.extend(candidate for candidate in QUOTED_NAME_RE.findall(text) if _looks_like_name_variant(candidate, known_names))

    for token in TOKEN_RE.findall(text):
        stem = _strip_particle(token)
        if stem and _looks_like_name_variant(stem, known_names):
            found.append(stem)

    issues: list[str] = []
    for candidate in found:
        if candidate in known_set:
            continue
        if any(candidate == f"{name}아" or candidate == f"{name}야" for name in known_names):
            continue
        reason = "unknown name"
        if _looks_like_name_variant(candidate, known_names):
            reason = "possible name variant"
        message = f"{reason}: '{candidate}' is not in allowed names ({', '.join(known_names)})"
        if message not in issues:
            issues.append(message)
    return ConsistencyCheck(known_names=known_names, issues=issues)


def repair_name_consistency(
    config: "AppConfig",
    client: "OllamaClient",
    text: str,
    world: str,
    characters: str,
    previous_scene: str,
) -> str:
    if not config.generation.enable_consistency_repair:
        return text
    check = check_name_consistency(text, characters)
    if not check.issues:
        return text

    prompt = f"""
다음 웹소설 본문에서 인물명 일관성 오류만 고쳐 다시 출력하세요.

[세계관]
{world}

[인물표]
{characters}

[이전 장면]
{previous_scene}

[허용 인물명]
{", ".join(check.known_names)}

[감지된 문제]
{chr(10).join(f"- {issue}" for issue in check.issues)}

[수정 규칙]
- 허용 인물명에 없는 고유명사 인물명은 만들지 않습니다.
- 잘못된 이름 변형은 가장 자연스러운 허용 인물명 또는 역할명으로 바꿉니다.
- 이름이 확정되지 않은 가족은 새 이름 대신 "동생"처럼 역할명으로 씁니다.
- 사건 순서, 문체, 분량은 최대한 유지합니다.
- 본문만 출력합니다.

[원문]
{text}
"""
    try:
        repaired = client.chat(
            prompt,
            system="당신은 한국어 장편 소설의 고유명사와 설정 일관성을 교정하는 편집자입니다.",
            temperature=0.2,
            max_tokens=config.generation.max_tokens,
        ).strip()
    except Exception:
        return text
    return repaired or text


def _strip_particle(token: str) -> str:
    for particle in PARTICLES:
        if token.endswith(particle) and len(token) > len(particle) + 1:
            return token[: -len(particle)]
    return token


def _looks_like_name_variant(candidate: str, known_names: list[str]) -> bool:
    if not (2 <= len(candidate) <= 4):
        return False
    for name in known_names:
        if candidate == name:
            return False
        if candidate == name[::-1]:
            return True
        if len(candidate) >= 3 and (candidate.startswith(name[::-1]) or candidate.endswith(name[::-1])):
            return True
    return False
