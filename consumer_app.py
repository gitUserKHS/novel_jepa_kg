"""이야기 공방 — 소비자 앱 (ChatGPT 식 두 모드: 일반 채팅 / 장편 소설).

- 일반 채팅: 로컬 Qwen 모델 서버에 직접 스트리밍. 대화는 계정별로 저장(ChatStore).
- 장편 소설: 자유로운 말로 기획 대화 → 작품 카드 확정 → 장 단위 연재(큐 + 워커).
"""

from __future__ import annotations

import html
import json
import math
import re
import time
from datetime import UTC, datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.llm.local_client import LocalLLMError
from src.service.chat_store import ChatStore, ChatStoreError
from src.service.consumer_store import (
    JOB_FAILED,
    JOB_FAILED_RECOVERABLE,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    AccountExistsError,
    ConsumerStore,
    ConsumerStoreError,
)
from src.service.runtime import make_llm_client
from src.service.story_workspace import (
    StoryWorkspace,
    build_continuation_bundle,
    read_draft,
    read_live_prose,
    split_sections,
)
from src.memory.story_outline import load_story_outline
from src.utils.config import AppConfig, load_config


st.set_page_config(page_title="이야기 공방", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

_SESSION_COOKIE = "novel_jepa_session"
MODE_CHAT = "💬 일반 채팅"
MODE_NOVEL = "📖 장편 소설"
CREATIVITY_LABELS = {"안정": "stable", "균형": "balanced", "대담": "bold"}
STATUS_LABELS = {
    JOB_QUEUED: "대기 중",
    JOB_RUNNING: "집필 중",
    JOB_SUCCEEDED: "완료",
    JOB_FAILED: "실패",
    JOB_FAILED_RECOVERABLE: "복구 가능",
}
TONE_OPTIONS = {"기본 말투": "", "밝고 귀여운 말투": "cute"}
CHAT_SYSTEM_PROMPT = (
    "당신은 친절하고 정확한 한국어 AI 어시스턴트다. 질문에는 간결하고 자연스러운 한국어로 답하고, "
    "모르는 것은 모른다고 말한다. 코드나 목록이 필요할 때만 마크다운을 쓴다."
)
PLANNER_SYSTEM_PROMPT = (
    "너는 한국어 장편 소설의 기획 편집자다. 사용자가 편하게 말한 바람(장르, 분위기, 주인공, 소재, 분량 등)을 듣고 "
    "작품 카드를 채운다. 사용자가 명시한 내용은 그대로 두고, 비어 있는 항목은 사용자의 말과 어울리게 스스로 제안해 채운다. "
    "title 은 매력적인 한국어 제목, genre 는 짧은 장르명, premise 는 두세 문장의 핵심 소재, world 는 서너 문장의 세계관, "
    "protagonist 는 '이름: 한 줄 소개', characters 는 '이름: 역할과 목표' 를 줄바꿈으로 나열(2~4명), "
    "target_chars 는 10000~50000 사이의 정수(기본 30000). "
    "reply 에는 사용자에게 건넬 한두 문장(제안 요약이나 확인 질문)을 쓴다. 모든 항목이 채워졌으면 ready 를 true 로 둔다. "
    "반드시 아래 형태의 JSON 객체 하나만 출력한다:\n"
    '{"reply": "...", "ready": true, "card": {"title": "...", "genre": "...", "premise": "...", "world": "...", '
    '"protagonist": "...", "characters": "...", "target_chars": 30000}}'
)
CARD_FIELDS = ("title", "genre", "premise", "world", "protagonist", "characters", "target_chars")
CARD_LABELS = {
    "title": "제목", "genre": "장르", "premise": "핵심 소재", "world": "세계관",
    "protagonist": "주인공", "characters": "주요 인물", "target_chars": "목표 분량",
}


# ---- 스타일 ----------------------------------------------------------------------------------

def _styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #202624; --muted: #68716d; --line: #d9dfdb; --jade: #0b7569;
            --jade-dark: #07594f; --coral: #c65c4b; --paper: #f7f8f5; --white: #ffffff;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        /* 헤더는 남긴다: 사이드바(모드 전환)를 접었다 펴는 버튼이 여기에 있다. */
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
        .block-container { max-width: 1100px; padding-top: 1.6rem; padding-bottom: 5rem; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        .stApp h1 { font-size: 1.8rem; line-height: 1.25; margin-bottom: 0.3rem; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important; border-radius: 10px !important; background: var(--white);
        }
        .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
            border-radius: 8px !important; min-height: 2.4rem;
        }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--jade) !important; border-color: var(--jade) !important; color: #fff !important;
        }
        [data-testid="stChatMessage"] { border-bottom: 1px solid var(--line); border-radius: 0; padding: 0.9rem 0.2rem; }
        [data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; }
        .story-kicker { color: var(--jade); font-size: 0.74rem; font-weight: 750; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
        .story-meta { color: var(--muted); font-size: 0.9rem; }
        .settings-block { border-left: 2px solid var(--line); padding: 0.2rem 0 0.2rem 0.8rem; margin: 0 0 0.8rem;
                          color: var(--muted); font-size: 0.92rem; white-space: pre-wrap; }
        .auth-copy { color: var(--muted); font-size: 1rem; margin: 0 0 1.4rem; }
        .auth-note { border-top: 1px solid var(--line); color: var(--muted); font-size: 0.86rem; margin-top: 1.2rem; padding-top: 1rem; }
        .auth-art-caption { color: var(--muted); font-size: 0.85rem; margin-top: 0.6rem; }
        .outline-strip { border-left: 3px solid var(--jade); background: #eef3ef; padding: 0.7rem 0.85rem; margin: 0.6rem 0 1rem; }
        .outline-label { color: var(--jade-dark); font-size: 0.76rem; font-weight: 750; }
        .outline-text { color: var(--ink); font-size: 0.93rem; margin-top: 0.15rem; }
        .queue-note { border-left: 4px solid var(--coral); padding: 0.3rem 0.8rem; color: var(--ink); }
        .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.2rem; }
        .card-item b { color: var(--jade-dark); font-size: 0.8rem; letter-spacing: 0.04em; }
        .card-item div { white-space: pre-wrap; font-size: 0.93rem; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 0.7rem 5rem; }
            .card-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- 런타임 / 세션 ---------------------------------------------------------------------------

def _load_runtime() -> tuple[AppConfig, ConsumerStore, ChatStore, Any]:
    config = load_config("configs/default.yaml")
    return config, ConsumerStore(config), ChatStore(config), make_llm_client(config)


def _model_status(client: Any) -> dict[str, Any]:
    cache_key = "consumer_model_status"
    cached = st.session_state.get(cache_key)
    now = time.time()
    if cached and now - cached["at"] < 5:
        return cached["status"]
    status = client.status()
    st.session_state[cache_key] = {"at": now, "status": status}
    return status


def _clear_story_session() -> None:
    for key in ("consumer_story_id", "consumer_last_job_status", "delete_story_confirmation"):
        st.session_state.pop(key, None)


def _clear_account_session() -> None:
    for key in tuple(st.session_state.keys()):
        if str(key).startswith(("consumer_", "plan_", "chat_")) or key == "delete_story_confirmation":
            st.session_state.pop(key, None)


def _set_browser_cookie(value: str, max_age: int) -> None:
    payload = json.dumps(f"{_SESSION_COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax")
    components.html(
        f"""<script>
        const cookie = {payload};
        try {{ window.parent.document.cookie = cookie; }}
        catch (error) {{ document.cookie = cookie; }}
        </script>""",
        height=0,
    )


def _sync_session_cookie(token: str, days: int) -> None:
    if st.session_state.get("consumer_cookie_synced") == token:
        return
    _set_browser_cookie(token, max(60, int(days) * 86400))
    st.session_state["consumer_cookie_synced"] = token


def _forget_session_cookie() -> None:
    _set_browser_cookie("", 0)
    st.session_state.pop("consumer_cookie_synced", None)


def _current_user(store: ConsumerStore) -> dict[str, Any] | None:
    token = str(st.session_state.get("consumer_session_token", ""))
    if not token:
        try:
            stored = st.context.cookies.get(_SESSION_COOKIE, "")
        except Exception:  # noqa: BLE001
            stored = ""
        token = stored if isinstance(stored, str) else ""
        if token:
            st.session_state["consumer_session_token"] = token
    if not token:
        return None
    user = store.authenticate_user_session(token)
    if user is None:
        _clear_account_session()
        _forget_session_cookie()
    return user


def _finish_login(store: ConsumerStore, user: dict[str, Any]) -> None:
    st.session_state["consumer_session_token"] = store.create_user_session(str(user["id"]))
    _clear_story_session()
    st.rerun()


def _auth_entry(config: AppConfig, store: ConsumerStore) -> None:
    auth, artwork = st.columns([0.86, 1.14], gap="large", vertical_alignment="center")
    with auth:
        with st.container(border=True):
            st.markdown('<div class="story-kicker">STORY STUDIO</div>', unsafe_allow_html=True)
            st.title(config.consumer.name)
            st.markdown('<div class="auth-copy">편하게 말하면, 이야기가 됩니다.</div>', unsafe_allow_html=True)
            login_tab, register_tab = st.tabs(["로그인", "회원가입"])
            with login_tab:
                with st.form("consumer_login_form", clear_on_submit=False, border=False):
                    username = st.text_input("아이디", max_chars=40, autocomplete="username")
                    password = st.text_input("비밀번호", type="password", max_chars=128, autocomplete="current-password")
                    submitted = st.form_submit_button("로그인", type="primary", width="stretch")
                if submitted:
                    user = store.authenticate_user(username, password)
                    if user is None:
                        st.error("아이디 또는 비밀번호가 맞지 않아.")
                    else:
                        _finish_login(store, user)
            with register_tab:
                with st.form("consumer_register_form", clear_on_submit=False, border=False):
                    display_name = st.text_input("표시 이름", max_chars=40)
                    username = st.text_input("새 아이디", max_chars=40, autocomplete="username",
                                             help="영문 또는 숫자로 시작하는 3~40자의 영문, 숫자, ., _, -")
                    password = st.text_input("새 비밀번호", type="password", max_chars=128, autocomplete="new-password")
                    password_confirm = st.text_input("비밀번호 확인", type="password", max_chars=128, autocomplete="new-password")
                    submitted = st.form_submit_button("회원가입", type="primary", width="stretch")
                if submitted:
                    if password != password_confirm:
                        st.error("비밀번호 확인이 일치하지 않아.")
                    else:
                        try:
                            user = store.create_user(username=username, display_name=display_name, password=password)
                            _finish_login(store, user)
                        except (ValueError, AccountExistsError) as exc:
                            st.error(str(exc))
            st.markdown('<div class="auth-note">대화와 작품은 계정별로 분리 보관되며 작품은 30일 뒤 자동 정리돼.</div>',
                        unsafe_allow_html=True)
    with artwork:
        st.image("assets/consumer/writing_studio.png", width="stretch")
        st.markdown('<div class="auth-art-caption">일반 채팅과 장편 소설 집필을 한 곳에서</div>', unsafe_allow_html=True)


def _logout(store: ConsumerStore) -> None:
    token = str(st.session_state.get("consumer_session_token", ""))
    if token:
        store.revoke_user_session(token)
    _clear_account_session()
    st.session_state["consumer_logout_pending"] = True
    st.rerun()


# ---- 사이드바 ---------------------------------------------------------------------------------

def _mode_bar(client: Any) -> str:
    """본문 상단의 모드 전환. 사이드바는 화면 폭에 따라 접히므로 여기에 둔다."""
    left, right = st.columns([1.6, 1], vertical_alignment="center")
    with left:
        mode = st.radio("모드", [MODE_CHAT, MODE_NOVEL], key="consumer_mode", horizontal=True,
                        label_visibility="collapsed")
    with right:
        status = _model_status(client)
        if status["ready"]:
            health = status.get("health") or {}
            st.caption(f"모델 준비됨 · {health.get('model', 'local')} · VRAM {health.get('vram_gib', '?')} GiB")
        else:
            st.caption("모델 서버 연결 안 됨 — run_model_server.bat 을 실행해줘.")
    return str(mode or MODE_CHAT)


def _sidebar(config: AppConfig, store: ConsumerStore, chats: ChatStore, user: dict[str, Any], mode: str) -> None:
    with st.sidebar:
        st.markdown('<div class="story-kicker">STORY STUDIO</div>', unsafe_allow_html=True)
        st.caption("대화 목록" if mode == MODE_CHAT else "작품 목록")
        if mode == MODE_CHAT:
            _sidebar_chats(chats, user)
        else:
            _sidebar_stories(store, user)
        st.divider()
        st.caption("로그인 계정")
        st.markdown(f"**{user['display_name']}**  \n@{user['username']}")
        if st.button("로그아웃", width="stretch", key="consumer_logout"):
            _logout(store)


def _sidebar_chats(chats: ChatStore, user: dict[str, Any]) -> None:
    if st.button("➕ 새 대화", width="stretch", key="chat_new", type="primary"):
        chat = chats.create_chat(str(user["id"]))
        st.session_state["chat_id"] = chat["id"]
        st.rerun()
    rows = chats.list_chats(str(user["id"]))
    active = str(st.session_state.get("chat_id", ""))
    if not rows:
        st.caption("아직 대화가 없어. 새 대화를 시작해봐.")
    for row in rows:
        label = ("▸ " if row["id"] == active else "") + str(row["title"])
        if st.button(label, key=f"chat_open_{row['id']}", width="stretch"):
            st.session_state["chat_id"] = row["id"]
            st.rerun()


def _sidebar_stories(store: ConsumerStore, user: dict[str, Any]) -> None:
    if st.button("➕ 새 작품 기획", width="stretch", key="story_new", type="primary"):
        _clear_story_session()
        st.session_state["plan_messages"] = []
        st.session_state["plan_card"] = {}
        st.rerun()
    stories = store.list_owned_stories(str(user["id"]))
    active = str(st.session_state.get("consumer_story_id", ""))
    if not stories:
        st.caption("아직 작품이 없어. 새 작품 기획에서 편하게 말해봐.")
    for story in stories:
        completed = bool(story.get("completed_at"))
        progress = 100.0 if completed else min(100.0, int(story["current_chars"]) / max(1, int(story["target_chars"])) * 100)
        marker = "▸ " if story["id"] == active else ""
        label = f"{marker}{story['title']}  ·  {progress:.0f}%" + (" 완결" if completed else "")
        if st.button(label, key=f"story_open_{story['id']}", width="stretch"):
            st.session_state["consumer_story_id"] = story["id"]
            st.rerun()


# ---- 일반 채팅 --------------------------------------------------------------------------------

def _chat_history_for_model(messages: list[dict[str, Any]], max_chars: int = 14000) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages):
        content = str(message["content"])
        if used + len(content) > max_chars and history:
            break
        history.append({"role": str(message["role"]), "content": content})
        used += len(content)
    history.reverse()
    return [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *history]


def _chat_mode(config: AppConfig, chats: ChatStore, client: Any, user: dict[str, Any]) -> None:
    user_id = str(user["id"])
    chat_id = str(st.session_state.get("chat_id", ""))
    chat = chats.get_chat(user_id, chat_id) if chat_id else None
    if chat is None:
        rows = chats.list_chats(user_id, limit=1)
        chat = rows[0] if rows else chats.create_chat(user_id, adapter=config.llm.chat_adapter)
        st.session_state["chat_id"] = chat["id"]
        chat_id = chat["id"]

    head, tone_col, clear_col = st.columns([3, 1.4, 0.8], vertical_alignment="center")
    with head:
        st.markdown('<div class="story-kicker">CHAT</div>', unsafe_allow_html=True)
        st.title(str(chat["title"]))
    with tone_col:
        current_tone = next((label for label, value in TONE_OPTIONS.items() if value == chat["adapter"]), "기본 말투")
        tone = st.selectbox("말투", list(TONE_OPTIONS), index=list(TONE_OPTIONS).index(current_tone), key=f"tone_{chat_id}")
        if TONE_OPTIONS[tone] != chat["adapter"]:
            chats.set_adapter(user_id, chat_id, TONE_OPTIONS[tone])
            chat["adapter"] = TONE_OPTIONS[tone]
    with clear_col:
        if st.button("🗑 삭제", key="chat_delete", width="stretch", help="이 대화를 완전히 지워."):
            chats.delete_chat(user_id, chat_id)
            st.session_state.pop("chat_id", None)
            st.rerun()

    messages = chats.list_messages(user_id, chat_id)
    if not messages:
        with st.chat_message("assistant"):
            st.markdown("무엇이든 물어봐. 소설을 쓰고 싶으면 왼쪽에서 **📖 장편 소설** 모드로 바꿔줘.")
    for message in messages:
        with st.chat_message(str(message["role"])):
            st.markdown(str(message["content"]))

    status = _model_status(client)
    blocked = "" if status["ready"] else f"모델 서버가 준비되지 않았어. {status['reason']}"
    if blocked:
        st.warning(blocked)
    prompt = st.chat_input("메시지를 입력해", disabled=bool(blocked), key=f"chat_input_{chat_id}")
    if not prompt:
        return
    try:
        chats.append_message(user_id, chat_id, "user", prompt)
    except ChatStoreError as exc:
        st.error(str(exc))
        return
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    history = _chat_history_for_model(messages)
    adapter = chat["adapter"] or None
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                client.stream_messages(
                    history,
                    temperature=config.llm.chat_temperature,
                    max_tokens=config.llm.chat_max_tokens,
                    adapter=adapter,
                    adapter_scale=config.llm.adapter_scale,
                )
            )
        except LocalLLMError as exc:
            st.error(str(exc))
            return
    reply_text = reply if isinstance(reply, str) else "".join(str(part) for part in reply)
    if reply_text.strip():
        chats.append_message(user_id, chat_id, "assistant", reply_text)
    st.rerun()


# ---- 장편 소설: 기획 대화 ---------------------------------------------------------------------

def _json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_target(config: AppConfig, value: Any) -> int:
    try:
        target = int(float(value))
    except (TypeError, ValueError):
        target = config.consumer.default_target_chars
    step = max(1, config.consumer.target_char_step)
    target = int(round(target / step) * step)
    return max(config.consumer.min_target_chars, min(config.consumer.max_target_chars, target))


def _merge_card(config: AppConfig, previous: dict[str, Any], incoming: Any) -> dict[str, Any]:
    card = dict(previous)
    if isinstance(incoming, dict):
        for field in CARD_FIELDS:
            value = incoming.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if field == "target_chars":
                card[field] = _normalize_target(config, value)
            elif isinstance(value, list):
                card[field] = "\n".join(str(item) for item in value)
            else:
                card[field] = str(value).strip()
    card.setdefault("target_chars", config.consumer.default_target_chars)
    return card


def _card_ready(card: dict[str, Any]) -> bool:
    return all(str(card.get(field, "")).strip() for field in ("title", "genre", "premise", "world", "protagonist"))


def _render_card(card: dict[str, Any]) -> None:
    items = []
    for field in CARD_FIELDS:
        value = card.get(field, "")
        if field == "target_chars":
            value = f"{int(value):,}자" if value else ""
        items.append(
            f'<div class="card-item"><b>{CARD_LABELS[field]}</b><div>{html.escape(str(value)) or "(아직 없음)"}</div></div>'
        )
    st.markdown('<div class="card-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def _plan_turn(config: AppConfig, client: Any, messages: list[dict[str, str]], card: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    context = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
    if card:
        context.append({"role": "system", "content": "지금까지 채운 작품 카드: " + json.dumps(card, ensure_ascii=False)})
    context.extend(messages[-8:])
    raw = client.chat_messages(context, temperature=0.5, max_tokens=900, json_mode=True, korean_filter=True)
    payload = _json_object(raw)
    if payload is None:
        return raw.strip(), card, False
    merged = _merge_card(config, card, payload.get("card"))
    reply = str(payload.get("reply") or "").strip() or "설정을 정리했어. 더 바꾸고 싶은 부분이 있으면 말해줘."
    ready = bool(payload.get("ready")) and _card_ready(merged)
    return reply, merged, ready


def _plan_mode(config: AppConfig, store: ConsumerStore, client: Any, user: dict[str, Any]) -> None:
    st.markdown('<div class="story-kicker">NEW STORY</div>', unsafe_allow_html=True)
    st.title("어떤 이야기를 쓰고 싶어?")
    st.markdown('<div class="story-meta">장르, 분위기, 주인공, 소재, 분량… 떠오르는 대로 편하게 말해줘. '
                '빈 곳은 내가 제안해서 채울게.</div>', unsafe_allow_html=True)
    messages: list[dict[str, str]] = st.session_state.setdefault("plan_messages", [])
    card: dict[str, Any] = st.session_state.setdefault("plan_card", {})

    if not messages:
        with st.chat_message("assistant"):
            st.markdown("예를 들면 이렇게: *\"비 오는 항구 도시에서 벌어지는 잔잔한 미스터리. 주인공은 등대지기 딸, 3만 자 정도.\"*")
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if card:
        with st.container(border=True):
            st.markdown("**작품 카드**")
            _render_card(card)
            ready = _card_ready(card)
            options = config.consumer.target_char_options()
            target = st.selectbox(
                "목표 분량", options=options,
                index=options.index(_normalize_target(config, card.get("target_chars"))),
                format_func=lambda value: f"{value:,}자", key="plan_target",
            )
            consent = st.checkbox("내 원고와 익명 품질 지표를 연구 개선에 활용하는 데 동의해", key="plan_consent")
            if not ready:
                st.caption("제목·장르·소재·세계관·주인공이 다 채워지면 시작할 수 있어. 대화로 보충해줘.")
            if st.button("✅ 이 설정으로 집필 시작", type="primary", disabled=not ready, key="plan_start", width="stretch"):
                try:
                    story = store.create_story(
                        str(user["id"]),
                        title=str(card.get("title", "")), genre=str(card.get("genre", "")),
                        premise=str(card.get("premise", "")), world=str(card.get("world", "")),
                        protagonist=str(card.get("protagonist", "")), characters=str(card.get("characters", "")),
                        target_chars=int(target), research_consent=bool(consent),
                    )
                    first_wish = next((m["content"] for m in messages if m["role"] == "user"), "")
                    instruction = "첫 장면을 시작해 줘." + (f" 참고: {first_wish[:300]}" if first_wish else "")
                    try:
                        store.enqueue_job(str(user["id"]), story["id"], instruction=instruction,
                                          creativity_profile="balanced", requested_chars=config.consumer.default_turn_chars)
                    except ConsumerStoreError:
                        pass
                    st.session_state.pop("plan_messages", None)
                    st.session_state.pop("plan_card", None)
                    st.session_state["consumer_story_id"] = story["id"]
                    st.rerun()
                except (ValueError, ConsumerStoreError) as exc:
                    st.error(str(exc))

    status = _model_status(client)
    blocked = "" if status["ready"] else f"모델 서버가 준비되지 않았어. {status['reason']}"
    if blocked:
        st.warning(blocked)
    prompt = st.chat_input("쓰고 싶은 이야기를 말해줘", disabled=bool(blocked), key="plan_input")
    if not prompt:
        return
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("작품 카드를 정리하는 중..."):
            try:
                reply, merged, _ready = _plan_turn(config, client, messages, card)
            except LocalLLMError as exc:
                st.error(str(exc))
                messages.pop()
                return
        st.markdown(reply)
    messages.append({"role": "assistant", "content": reply})
    st.session_state["plan_card"] = merged
    st.rerun()


# ---- 장편 소설: 집필 화면 ---------------------------------------------------------------------

def _worker_state(store: ConsumerStore) -> tuple[str, bool]:
    state = store.get_state("worker_heartbeat")
    if state is None:
        return "연결 대기", False
    try:
        payload = json.loads(state["value"])
        heartbeat = datetime.fromisoformat(state["updated_at"])
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=UTC)
        fresh = (datetime.now(UTC) - heartbeat).total_seconds() <= 20
        labels = {
            "idle": "준비됨", "polling": "준비됨", "running": "집필 중", "waiting_for_gpu": "자원 대기",
            "maintenance": "점검 중", "model_unavailable": "모델 준비 중",
        }
        return labels.get(str(payload.get("status", "unknown")), str(payload.get("status", "unknown"))), fresh
    except (ValueError, TypeError, json.JSONDecodeError):
        return "상태 확인 필요", False


def _typewriter(job_id: int, live_prose: str) -> None:
    key = f"consumer_typed_{job_id}"
    shown = str(st.session_state.get(key, ""))
    if not live_prose.startswith(shown):
        shown = ""
    delta = live_prose[len(shown):]
    placeholder = st.empty()
    if delta:
        steps = min(len(delta), 16)
        chunk = math.ceil(len(delta) / steps)
        pause = 0.64 / steps
        text = shown
        for offset in range(0, len(delta), chunk):
            text += delta[offset : offset + chunk]
            placeholder.markdown(text + " ▌")
            time.sleep(pause)
    placeholder.markdown(live_prose + " ▌")
    st.session_state[key] = live_prose


def _render_job(job: dict[str, Any], sections: list[str], store: ConsumerStore, user_id: str,
                story_id: str = "", live_prose: str = "") -> None:
    status = str(job["status"])
    job_id = int(job["id"])
    with st.chat_message("user"):
        instruction_column, delete_column = st.columns([9, 1], vertical_alignment="top")
        instruction_column.markdown(str(job["instruction"]))
        if story_id and status not in (JOB_QUEUED, JOB_RUNNING):
            if delete_column.button("✕", key=f"delete_job_{job_id}", help="이 턴과 이 턴이 쓴 원고를 함께 지워."):
                try:
                    store.delete_owned_job(user_id, story_id, job_id)
                except ConsumerStoreError as exc:
                    st.error(str(exc))
                else:
                    st.rerun(scope="fragment")
    with st.chat_message("assistant"):
        if status == JOB_SUCCEEDED:
            st.session_state.pop(f"consumer_typed_{job_id}", None)
            start = int(job["start_section_count"])
            count = int(job["result_section_count"])
            generated = sections[start : start + count]
            if generated:
                for section in generated:
                    st.markdown(section)
            else:
                st.caption("이 턴의 원고는 전체 원고 파일에 저장되어 있어.")
        elif status == JOB_QUEUED:
            position = store.queue_position(user_id, job_id)
            st.markdown(f'<div class="queue-note">대기 순번 {position or "-"}번 · 요청을 안전하게 보관했어.</div>',
                        unsafe_allow_html=True)
        elif status == JOB_RUNNING:
            start = int(job["start_section_count"])
            committed = sections[start:]
            for section in committed:
                st.markdown(section)
            if live_prose:
                _typewriter(job_id, live_prose)
                st.caption("✍️ 집필하는 중이야. 브라우저를 닫아도 작업은 계속돼.")
            elif committed:
                st.caption("✍️ 다음 장을 준비하고 있어. 브라우저를 닫아도 작업은 계속돼.")
            else:
                st.info("이야기 지도를 그리고 첫 장을 쓰는 중이야. 브라우저를 닫아도 작업은 계속돼.")
        elif status == JOB_FAILED_RECOVERABLE:
            st.warning(str(job.get("error_public") or "저장된 장부터 다시 이어 쓸 수 있어."))
        else:
            st.error(str(job.get("error_public") or "생성하지 못했어. 다시 요청해줘."))


def _settings_view(story: dict[str, Any]) -> None:
    st.markdown(f"**제목** · {story['title']}")
    st.markdown(f"**장르** · {story['genre']}")
    st.caption(f"목표 분량 {int(story['target_chars']):,}자")
    for label, key in (("핵심 소재", "premise"), ("세계관", "world"), ("주인공", "protagonist"), ("주요 인물", "characters")):
        value = str(story[key] or "").strip()
        st.markdown(f"**{label}**")
        st.markdown(f'<div class="settings-block">{html.escape(value) or "(비어 있음)"}</div>', unsafe_allow_html=True)
    st.caption("연구 활용 동의: " + ("동의함" if int(story["research_consent"] or 0) else "동의하지 않음"))


def _settings_form(config: AppConfig, store: ConsumerStore, user_id: str, story: dict[str, Any]) -> None:
    story_id = str(story["id"])
    with st.form(f"edit_story_{story_id}"):
        title_column, genre_column = st.columns([1.2, 1])
        title = title_column.text_input("제목", value=str(story["title"]), max_chars=100)
        genre = genre_column.text_input("장르", value=str(story["genre"]), max_chars=80)
        premise = st.text_area("핵심 소재", value=str(story["premise"]), height=100, max_chars=1500)
        world = st.text_area("세계관", value=str(story["world"]), height=140, max_chars=4000)
        protagonist_column, characters_column = st.columns(2)
        protagonist = protagonist_column.text_area("주인공", value=str(story["protagonist"]), height=120, max_chars=2000)
        characters = characters_column.text_area("주요 인물 (선택)", value=str(story["characters"] or ""), height=120, max_chars=3000)
        current_chars = int(story["current_chars"])
        options = [value for value in config.consumer.target_char_options() if value >= current_chars]
        current_target = int(story["target_chars"])
        if current_target not in options:
            options = sorted({*options, current_target})
        target_chars = st.selectbox("전체 목표 글자 수", options=options, index=options.index(current_target),
                                    format_func=lambda value: f"{value:,}자", help="이미 쓴 분량보다 작게 줄일 수는 없어.")
        consent = st.checkbox("내 원고와 익명 품질 지표를 연구 개선에 활용하는 데 동의해",
                              value=bool(int(story["research_consent"] or 0)))
        saved = st.form_submit_button("설정 저장", type="primary", width="stretch")
    if saved:
        try:
            store.update_owned_story(user_id, story_id, title=title, genre=genre, premise=premise, world=world,
                                     protagonist=protagonist, characters=characters, target_chars=int(target_chars),
                                     research_consent=consent)
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))
        else:
            st.success("설정을 저장했어. 다음 턴부터 반영돼.")
            st.rerun()


@st.fragment(run_every=1.0)
def _story_live(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story_id: str) -> None:
    story = store.get_owned_story(user_id, story_id)
    if story is None:
        st.error("작품을 찾을 수 없거나 보관 기간이 끝났어.")
        return
    workspace = StoryWorkspace.for_story(config, story_id, create=True)
    draft = read_draft(workspace.draft)
    sections = split_sections(draft)
    live_prose = read_live_prose(workspace)
    jobs = list(reversed(store.list_owned_jobs(user_id, story_id)))
    outstanding = store.owned_outstanding_job(user_id, story_id)
    queue = store.queue_stats()
    worker_label, worker_fresh = _worker_state(store)
    model = _model_status(client)

    completed = bool(story.get("completed_at"))
    written_chars = len(draft) + len(live_prose)
    progress = 1.0 if completed else min(1.0, written_chars / max(1, int(story["target_chars"])))
    outline = load_story_outline(workspace.outline)
    if outline and outline.beats:
        active_beat = outline.beats[min(len(outline.beats) - 1, int(progress * len(outline.beats)))]
        st.markdown(
            '<div class="outline-strip">'
            f'<div class="outline-label">현재 흐름 · {html.escape(active_beat.phase)}</div>'
            f'<div class="outline-text">{html.escape(active_beat.purpose)}</div></div>',
            unsafe_allow_html=True,
        )
    metric_columns = st.columns(4)
    metric_columns[0].metric("진행률", f"{progress * 100:.1f}%")
    metric_columns[1].metric("분량", f"{written_chars:,} / {int(story['target_chars']):,}자")
    metric_columns[2].metric("장", f"{len(sections)}개")
    metric_columns[3].metric("집필 상태", "완결" if completed else (worker_label if worker_fresh else "연결 확인 중"))
    st.progress(progress)

    if jobs:
        clearable = [job for job in jobs if str(job["status"]) not in (JOB_QUEUED, JOB_RUNNING)]
        if clearable:
            header_spacer, clear_column = st.columns([3, 1], vertical_alignment="center")
            header_spacer.caption(f"대화 {len(clearable)}개")
            if clear_column.button(f"대화 전체 삭제 ({len(clearable)})", key="clear_job_history",
                                   help="끝난 턴과 그 턴들이 쓴 원고를 모두 지워. 집필 중인 턴은 남아.", width="stretch"):
                try:
                    store.clear_owned_job_history(user_id, story_id)
                except ConsumerStoreError as exc:
                    st.error(str(exc))
                else:
                    st.rerun(scope="fragment")
        for job in jobs:
            _render_job(job, sections, store, user_id, story_id=story_id, live_prose=live_prose)
    else:
        with st.chat_message("assistant"):
            st.markdown("첫 장면에서 일어날 사건이나 원하는 분위기를 말해줘.")

    st.divider()
    control_a, control_b = st.columns([1.3, 1])
    with control_a:
        creativity_label = st.segmented_control("창의성", options=list(CREATIVITY_LABELS), default="균형",
                                                selection_mode="single", key="consumer_creativity")
    with control_b:
        turn_chars = st.selectbox("이번에 생성할 글자 수", options=config.consumer.allowed_turn_chars,
                                  index=config.consumer.allowed_turn_chars.index(config.consumer.default_turn_chars),
                                  format_func=lambda value: f"약 {value:,}자", key="consumer_turn_chars")

    blocked_reason = ""
    if store.is_maintenance():
        blocked_reason = "서비스 점검 중이야. 진행 중인 원고는 안전하게 보관돼."
    elif not model["ready"]:
        blocked_reason = f"집필 모델을 준비하고 있어. {model['reason']}"
    elif outstanding:
        blocked_reason = f"현재 요청이 {STATUS_LABELS.get(str(outstanding['status']), '처리 중')}이야."
    elif not worker_fresh:
        blocked_reason = "집필 worker 연결을 확인하고 있어. (run_service.bat 이 실행 중인지 확인해줘)"
    elif completed:
        blocked_reason = "결말까지 완성했어. 전체 원고나 이어쓰기 번들을 내려받을 수 있어."
    if blocked_reason:
        st.caption(blocked_reason)
    elif len(draft) >= int(story["target_chars"]):
        st.caption("목표 분량에 도달했어. 다음 요청에서 결말 장면까지 완성할게.")

    prompt = st.chat_input("다음 전개를 말해줘 (예: 박 노인이 20년 전 일을 털어놓게 해줘)",
                           disabled=bool(blocked_reason), key="consumer_chat_input")
    if prompt:
        try:
            store.enqueue_job(user_id, story_id, instruction=prompt,
                              creativity_profile=CREATIVITY_LABELS.get(str(creativity_label), "balanced"),
                              requested_chars=int(turn_chars))
            st.rerun(scope="fragment")
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))

    if queue["queued"] or queue["running"]:
        st.caption(f"전체 대기 {queue['queued']}건 · 실행 {queue['running']}건")

    st.divider()
    download_a, download_b = st.columns(2)
    download_a.download_button("전체 원고", data=draft.encode("utf-8"), file_name=f"{story['title']}.md",
                               mime="text/markdown", disabled=not bool(draft), width="stretch")
    download_b.download_button("이어쓰기 번들", data=build_continuation_bundle(workspace, story),
                               file_name=f"{story['title']}_continue.zip", mime="application/zip", width="stretch")

    with st.expander("작품 관리 · 초기화와 삭제"):
        st.caption("되돌릴 수 없어. 먼저 위에서 원고를 내려받아 두는 걸 권해.")
        reset_column, delete_column = st.columns(2)
        with reset_column:
            st.markdown("**원고 초기화**")
            st.caption("제목·장르·세계관·인물 설정은 남기고 본문과 기억을 모두 지워.")
            reset_ready = st.checkbox("원고를 지우고 처음부터 다시 쓸게", key="reset_story_confirmation")
            if st.button("원고 초기화", disabled=not reset_ready, width="stretch", key="reset_story_button"):
                try:
                    store.reset_owned_story(user_id, story_id)
                except ConsumerStoreError as exc:
                    st.error(str(exc))
                else:
                    st.rerun(scope="fragment")
        with delete_column:
            st.markdown("**작품 삭제**")
            st.caption("작품과 원고를 데이터베이스에서 완전히 지우고 목록에서 없애.")
            delete_ready = st.checkbox("이 작품을 완전히 삭제할게", key="delete_story_main_confirmation")
            if st.button("작품 삭제", disabled=not delete_ready, width="stretch", key="delete_story_main_button"):
                try:
                    store.delete_owned_story(user_id, story_id)
                except ConsumerStoreError as exc:
                    st.error(str(exc))
                else:
                    _clear_story_session()
                    st.rerun()


def _story_mode(config: AppConfig, store: ConsumerStore, client: Any, user: dict[str, Any], story: dict[str, Any]) -> None:
    back_column, _spacer = st.columns([1, 5])
    if back_column.button("← 새 작품 기획", key="story_back_to_plan", width="stretch"):
        _clear_story_session()
        st.session_state["plan_messages"] = []
        st.session_state["plan_card"] = {}
        st.rerun()
    st.markdown('<div class="story-kicker">IN PROGRESS</div>', unsafe_allow_html=True)
    st.title(str(story["title"]))
    st.markdown(f'<div class="story-meta">{html.escape(str(story["genre"]))} · 장을 이어 쓰며 완성하는 장편</div>',
                unsafe_allow_html=True)
    with st.expander("작품 설정 · 세계관과 인물"):
        view_tab, edit_tab = st.tabs(["설정 보기", "설정 수정"])
        with view_tab:
            _settings_view(story)
        with edit_tab:
            st.caption("세계관과 인물은 매 장 프롬프트에 다시 들어가. 수정하면 다음 턴부터 반영되고, 이미 쓴 본문은 그대로 남아.")
            _settings_form(config, store, str(user["id"]), story)
    _story_live(config, store, client, str(user["id"]), str(story["id"]))


# ---- 메인 --------------------------------------------------------------------------------------

def main() -> None:
    _styles()
    config, store, chats, client = _load_runtime()
    if st.session_state.pop("consumer_logout_pending", False):
        _forget_session_cookie()
    user = _current_user(store)
    if user is None:
        _auth_entry(config, store)
        return
    _sync_session_cookie(str(st.session_state.get("consumer_session_token", "")), config.consumer.auth_session_days)

    mode = _mode_bar(client)
    _sidebar(config, store, chats, user, mode)
    if mode == MODE_CHAT:
        _chat_mode(config, chats, client, user)
        return
    story_id = str(st.session_state.get("consumer_story_id", ""))
    story = store.get_owned_story(str(user["id"]), story_id) if story_id else None
    if story is None:
        _clear_story_session()
        _plan_mode(config, store, client, user)
        return
    _story_mode(config, store, client, user, story)


if __name__ == "__main__":
    main()
