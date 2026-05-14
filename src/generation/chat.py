from __future__ import annotations

from typing import Any

from src.embedding.vector_store import retrieve_by_text, retrieve_by_vector
from src.generation.consistency import allowed_name_instruction, build_beat_card, repair_name_consistency
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import prose_prompt
from src.memory.context import (
    build_memory_prompt,
    compress_session_memory,
    extract_knowledge_graph,
    should_compress,
    summarize_scene,
)
from src.planner.predict import predict_next_embedding
from src.session.store import append_message, append_scene_summary, save_session
from src.utils.config import AppConfig


CHAT_MODES = ["LLM only", "RAG + LLM", "JEPA Planner + RAG + LLM"]


def _previous_scene_for_retrieval(session: dict[str, Any], user_instruction: str) -> str:
    current = session.get("story_state", {}).get("current_scene", "")
    if current:
        return current
    scenes = session.get("scene_summaries", [])
    if scenes:
        return scenes[-1].get("summary", user_instruction)
    return user_instruction


def _retrieve_examples(
    config: AppConfig,
    client: OllamaClient,
    session: dict[str, Any],
    user_instruction: str,
    mode: str,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if mode == "LLM only":
        return "세션 메모리를 바탕으로 다음 장면을 자연스럽게 전개한다.", [], []

    previous_scene = _previous_scene_for_retrieval(session, user_instruction)
    try:
        if mode == "RAG + LLM":
            retrieved = retrieve_by_text(config, client, previous_scene, config.generation.top_k)
            examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in retrieved[: config.generation.rag_context_limit]]
            return "검색된 유사 장면의 전환 논리를 참고해 다음 갈등을 확장한다.", examples, retrieved

        predicted = predict_next_embedding(config, client, previous_scene)
        retrieved = retrieve_by_vector(config, predicted, config.generation.top_k)
        examples = [item["sample"]["scene_t_plus_1"]["summary"] for item in retrieved[: config.generation.rag_context_limit]]
        direction = examples[0] if examples else "예측된 다음 장면 방향에 맞춰 갈등을 한 단계 진전시킨다."
        return direction, examples, retrieved
    except Exception as exc:  # noqa: BLE001 - Chat should degrade instead of losing the session.
        return f"검색/예측 메모리 사용 실패: {exc}. 세션 메모리만으로 다음 장면을 전개한다.", [], []


def generate_chat_turn(
    config: AppConfig,
    client: OllamaClient,
    session: dict[str, Any],
    user_instruction: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in CHAT_MODES:
        raise ValueError(f"Unknown chat generation mode: {mode}")
    user_instruction = user_instruction.strip()
    if not user_instruction:
        raise ValueError("User instruction is empty.")

    append_message(session, "user", user_instruction, mode=mode)
    memory_context = build_memory_prompt(config, session, user_instruction)
    direction, examples, retrieved = _retrieve_examples(config, client, session, user_instruction, mode)
    prompt = prose_prompt(
        world=session.get("world", ""),
        characters=session.get("characters", ""),
        previous_scene=memory_context,
        style=config.generation.style,
        direction=direction,
        examples=examples,
        beat_card=build_beat_card(mode, direction, examples, session.get("characters", ""), config.generation.rag_context_limit),
        consistency_rules=allowed_name_instruction(session.get("characters", "")),
    )
    assistant_text = client.chat(
        prompt,
        system="당신은 장편 한국어 웹소설을 세션 메모리와 설정에 맞춰 이어 쓰는 작가입니다.",
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    ).strip()
    assistant_text = repair_name_consistency(
        config,
        client,
        assistant_text,
        session.get("world", ""),
        session.get("characters", ""),
        memory_context,
    )
    append_message(
        session,
        "assistant",
        assistant_text,
        mode=mode,
        metadata={
            "retrieved_count": len(retrieved),
            "direction": direction,
        },
    )
    scene_summary = summarize_scene(config, client, assistant_text)
    append_scene_summary(session, scene_summary, mode)

    graph = session.get("knowledge_graph", {"nodes": [], "edges": []})
    if config.chat.auto_update_graph:
        graph = extract_knowledge_graph(config, client, session, f"{user_instruction}\n\n{assistant_text}\n\n{scene_summary}")

    compressed = False
    if should_compress(config, session):
        compress_session_memory(config, client, session)
        compressed = True

    save_session(config, session)
    return {
        "session": session,
        "assistant_text": assistant_text,
        "scene_summary": scene_summary,
        "retrieved": retrieved,
        "direction": direction,
        "compressed": compressed,
        "graph": graph,
    }
