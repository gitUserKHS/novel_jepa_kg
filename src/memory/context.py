from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from src.data.validate_jsonl import parse_json_object
from src.llm.ollama_client import OllamaClient
from src.utils.config import AppConfig


def total_message_chars(session: dict[str, Any]) -> int:
    return sum(len(message.get("content", "")) for message in session.get("messages", []))


def recent_messages(session: dict[str, Any], count: int) -> list[dict[str, Any]]:
    return session.get("messages", [])[-count:]


def should_compress(config: AppConfig, session: dict[str, Any]) -> bool:
    message_count = len(session.get("messages", []))
    if message_count and message_count % max(1, config.chat.compress_every_messages) == 0:
        return True
    return total_message_chars(session) >= config.chat.compress_over_chars


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def summarize_scene(config: AppConfig, client: OllamaClient, text: str) -> str:
    if not text.strip():
        return ""
    if client.dry_run:
        return _clip(re.sub(r"\s+", " ", text), config.chat.scene_summary_chars)
    prompt = f"""
다음 웹소설 본문을 장면 요약으로 압축하세요.

조건:
- 한국어.
- {config.chat.scene_summary_chars}자 이내.
- 사건 변화, 새 단서, 감정 변화, 다음 선택지를 포함.
- 본문을 다시 쓰지 말고 요약만 출력.

[본문]
{text}
"""
    return client.chat(prompt, system="당신은 장편 소설의 장면 요약을 관리하는 편집자입니다.", temperature=0.2, max_tokens=500).strip()


def compress_session_memory(config: AppConfig, client: OllamaClient, session: dict[str, Any]) -> str:
    messages = session.get("messages", [])
    if not messages:
        return session.get("memory_summary", "")
    transcript = "\n".join(f"{message.get('role')}: {message.get('content')}" for message in messages[-24:])
    previous = session.get("memory_summary", "")
    scenes = "\n".join(scene.get("summary", "") for scene in session.get("scene_summaries", [])[-12:])
    if client.dry_run:
        summary = "\n".join(part for part in [previous, scenes, _clip(transcript, 1500)] if part)
        session["memory_summary"] = _clip(summary, config.chat.max_memory_chars)
        return session["memory_summary"]
    prompt = f"""
장편 소설 채팅 세션의 장기 기억을 갱신하세요.

유지해야 할 정보:
- 세계관 규칙과 금지 사항
- 인물 목표, 비밀, 관계 변화
- 사건의 원인과 결과
- 회수되지 않은 떡밥과 질문
- 현재 장면 상태와 다음 선택지

출력 조건:
- 한국어.
- {config.chat.max_memory_chars}자 이내.
- 섹션: 세계관, 인물, 사건 진행, 미해결 떡밥, 문체/주의.
- 이전 요약을 새 정보로 갱신하되 중복은 제거.

[이전 장기 기억]
{previous or "(없음)"}

[최근 장면 요약]
{scenes or "(없음)"}

[최근 대화]
{transcript}
"""
    summary = client.chat(prompt, system="당신은 장편 소설의 continuity editor입니다.", temperature=0.2, max_tokens=1800).strip()
    session["memory_summary"] = _clip(summary, config.chat.max_memory_chars)
    return session["memory_summary"]


def _normalize_id(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z가-힣_-]+", "", value)
    return value[:60] or "unknown"


def normalize_graph(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes = []
    edges = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        label = str(node.get("label") or node.get("id") or "").strip()
        if not label:
            continue
        nodes.append(
            {
                "id": _normalize_id(str(node.get("id") or label)),
                "label": label,
                "type": str(node.get("type") or "concept"),
                "summary": str(node.get("summary") or ""),
            }
        )
    node_ids = {node["id"] for node in nodes}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = _normalize_id(str(edge.get("source") or ""))
        target = _normalize_id(str(edge.get("target") or ""))
        if not source or not target:
            continue
        if source not in node_ids:
            nodes.append({"id": source, "label": source, "type": "concept", "summary": ""})
            node_ids.add(source)
        if target not in node_ids:
            nodes.append({"id": target, "label": target, "type": "concept", "summary": ""})
            node_ids.add(target)
        edges.append(
            {
                "source": source,
                "target": target,
                "type": str(edge.get("type") or "related_to"),
                "summary": str(edge.get("summary") or ""),
            }
        )
    return {"nodes": nodes, "edges": edges}


def merge_graphs(old_graph: dict[str, Any], new_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    old = normalize_graph(old_graph or {})
    new = normalize_graph(new_graph or {})
    nodes: dict[str, dict[str, Any]] = {node["id"]: node for node in old["nodes"]}
    for node in new["nodes"]:
        current = nodes.get(node["id"], {})
        nodes[node["id"]] = {
            **current,
            **node,
            "summary": node.get("summary") or current.get("summary", ""),
        }
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in old["edges"] + new["edges"]:
        key = (edge["source"], edge["target"], edge["type"])
        edge_map[key] = edge
    return {"nodes": list(nodes.values()), "edges": list(edge_map.values())}


def _dry_graph_from_text(text: str) -> dict[str, list[dict[str, Any]]]:
    names = re.findall(r"[가-힣]{2,4}", text)
    common = {"그리고", "하지만", "장면", "기억", "사실", "선택", "다음", "본문", "요약"}
    labels = []
    for name in names:
        if name not in common and name not in labels:
            labels.append(name)
        if len(labels) >= 6:
            break
    nodes = [{"id": _normalize_id(label), "label": label, "type": "character_or_concept", "summary": ""} for label in labels]
    edges = []
    for left, right in zip(labels, labels[1:]):
        edges.append({"source": _normalize_id(left), "target": _normalize_id(right), "type": "관련", "summary": "최근 장면에서 함께 언급됨"})
    return {"nodes": nodes, "edges": edges}


def extract_knowledge_graph(config: AppConfig, client: OllamaClient, session: dict[str, Any], new_text: str) -> dict[str, Any]:
    if not new_text.strip():
        return session.get("knowledge_graph", {"nodes": [], "edges": []})
    if client.dry_run:
        graph = _dry_graph_from_text(new_text)
        merged = merge_graphs(session.get("knowledge_graph", {}), graph)
        session["knowledge_graph"] = merged
        return merged
    prompt = f"""
다음 소설 세션 정보에서 지식 그래프를 JSON으로 추출하세요.

출력 JSON 스키마:
{{
  "nodes": [
    {{"id": "짧은_고유_ID", "label": "표시명", "type": "character|place|object|faction|clue|goal|conflict|rule", "summary": "한 줄 설명"}}
  ],
  "edges": [
    {{"source": "노드_ID", "target": "노드_ID", "type": "relationship|owns|seeks|hides|located_at|conflicts_with|causes", "summary": "한 줄 설명"}}
  ]
}}

조건:
- JSON 외의 설명 금지.
- 인물, 장소, 단서, 목표, 갈등, 세계관 규칙을 우선 추출.
- 이전 그래프와 합칠 수 있도록 안정적인 id를 사용.

[기존 장기 기억]
{session.get("memory_summary", "")}

[새 텍스트]
{new_text}
"""
    try:
        payload = parse_json_object(client.chat(prompt, system="당신은 소설 설정을 지식 그래프로 구조화하는 도우미입니다.", temperature=0.1, max_tokens=1200))
        merged = merge_graphs(session.get("knowledge_graph", {}), payload)
        session["knowledge_graph"] = merged
        return merged
    except Exception:
        return session.get("knowledge_graph", {"nodes": [], "edges": []})


def graph_tables(graph: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = normalize_graph(graph or {})
    return pd.DataFrame(graph["nodes"]), pd.DataFrame(graph["edges"])


def graph_to_mermaid(graph: dict[str, Any]) -> str:
    graph = normalize_graph(graph or {})
    lines = ["flowchart TD"]
    if not graph["nodes"]:
        return "flowchart TD\n  Empty[No graph nodes yet]"
    for node in graph["nodes"]:
        label = node["label"].replace('"', "'")
        lines.append(f'  {node["id"]}["{label}"]')
    for edge in graph["edges"]:
        edge_label = edge["type"].replace('"', "'")
        lines.append(f'  {edge["source"]} -- "{edge_label}" --> {edge["target"]}')
    return "\n".join(lines)


def build_memory_prompt(config: AppConfig, session: dict[str, Any], user_instruction: str) -> str:
    graph = normalize_graph(session.get("knowledge_graph", {}))
    node_lines = [f"- {node['label']} ({node['type']}): {node.get('summary', '')}" for node in graph["nodes"][:30]]
    edge_lines = [
        f"- {edge['source']} --{edge['type']}--> {edge['target']}: {edge.get('summary', '')}"
        for edge in graph["edges"][:40]
    ]
    recent = "\n".join(
        f"{message.get('role')}: {message.get('content')}"
        for message in recent_messages(session, config.chat.recent_messages)
    )
    scenes = "\n".join(
        f"- Scene {scene.get('index')}: {scene.get('summary')}"
        for scene in session.get("scene_summaries", [])[-8:]
    )
    story_state = json.dumps(session.get("story_state", {}), ensure_ascii=False, indent=2)
    context = f"""
[세션 제목]
{session.get('title', '')}

[세계관]
{session.get('world', '')}

[인물 설정]
{session.get('characters', '')}

[장기 기억 요약]
{session.get('memory_summary', '') or '(아직 없음)'}

[최근 장면 요약]
{scenes or '(아직 없음)'}

[지식 그래프 노드]
{chr(10).join(node_lines) or '(아직 없음)'}

[지식 그래프 엣지]
{chr(10).join(edge_lines) or '(아직 없음)'}

[현재 상태]
{story_state}

[최근 대화]
{recent or '(아직 없음)'}

[사용자 요청]
{user_instruction}
"""
    return _clip(context, config.chat.max_memory_chars + 4000)
