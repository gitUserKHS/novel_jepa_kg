"""토큰 상한에 잘린 JSON 객체 복구. 모델이 낸 JSON 을 읽는 모든 곳(메모리·설계·검토·퇴고 계획)이 쓴다."""

from __future__ import annotations

import json
import re
from typing import Any


def json_object(text: str) -> dict[str, Any] | None:
    """텍스트에서 JSON 객체 하나를 읽는다. 펜스·앞뒤 설명을 무시하고, 잘렸으면 마지막 완결 항목까지 살린다."""
    cleaned = re.sub(r"```[a-zA-Z]*", "", str(text or "")).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return salvage_json_object(cleaned)


def salvage_json_object(text: str) -> dict[str, Any] | None:
    """토큰 상한에 잘린 JSON 객체를 마지막 완결 항목까지 살려 닫는다.

    괄호 스택을 추적하며 쉼표·닫는 괄호 위치를 '안전 지점'으로 모은 뒤, 뒤에서부터
    하나씩 잘라 닫아 보고 처음 파싱되는 후보를 돌려준다. 값 없이 끝난 키나 미완성
    문자열은 자연히 건너뛴다.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    try:
        payload = json.loads(body[: body.rfind("}") + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_string = False
    escape = False
    safe_points: list[tuple[int, list[str]]] = []
    for index, char in enumerate(body):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
            safe_points.append((index + 1, list(stack)))
        elif char == ",":
            safe_points.append((index, list(stack)))
    for cut, open_brackets in reversed(safe_points[-80:]):
        candidate = body[:cut].rstrip().rstrip(",")
        candidate = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", candidate).rstrip().rstrip(",")
        candidate += "".join(reversed(open_brackets))
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload:
            return prune_incomplete_items(payload)
    return None


def prune_incomplete_items(payload: dict[str, Any]) -> dict[str, Any]:
    """잘린 꼬리에서 살아남은 불완전 객체(키가 모자란 항목)를 목록에서 뺀다."""
    for key, value in list(payload.items()):
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            widest = max(len(item) for item in value)
            payload[key] = [item for item in value if len(item) == widest]
    return payload
