"""이야기 공방 — 소비자 앱 (장편 소설 전용).

- 기획: 자유로운 말로 기획 대화 → 작품 카드 확정, 또는 양식 직접 작성(템플릿 저장·불러오기).
- 집필: 장 단위 연재(SQLite 큐 + 워커). 다음 전개를 말하거나 자동 이어쓰기에 맡긴다. 쓰는 장은 실시간으로 보이고,
  게이트·개연성 검토가 한 일(걷어내기·다시 쓰기·고쳐 쓰기·점수)이 턴 아래 남는다.
- 편집: 장 단위 직접 수정 / AI 퇴고 / 요약 메모리 편집.
- 모델: src/service/runtime.make_llm_client (기본 Ollama Gemma4 26B-A4B). 일반 채팅은 2026-09-03 에 제거했다.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
from datetime import UTC, datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

from src.llm.common import LocalLLMError
from src.service.consumer_store import (
    JOB_FAILED,
    JOB_FAILED_RECOVERABLE,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_ORIGIN_AUTO,
    JOB_SUCCEEDED,
    AccountExistsError,
    ConsumerStore,
    ConsumerStoreError,
)
from src.service.auto_continue import AUTO_PLACEHOLDER
from src.service.runtime import make_llm_client
from src.service.story_editor import (
    QUICK_REWRITES,
    StoryEditError,
    length_bounds,
    list_sections,
    normalize_edit,
    plan_edits,
    refresh_section_memory,
    replace_section,
    retry_note_for,
    section_memory,
    stream_rewrite,
    update_section_memory,
)
from src.service.template_store import (
    FIELD_LIMITS,
    TEMPLATE_FIELDS,
    TEMPLATE_NAME_CHARS,
    StoryTemplateStore,
    TemplateStoreError,
    template_from_story,
)
from src.service.story_workspace import (
    StoryWorkspace,
    build_continuation_bundle,
    read_draft,
    read_live_note,
    read_live_prose,
    split_sections,
)
from src.memory.story_outline import load_story_outline
from src.utils.config import AppConfig, load_config
from src.utils.timefmt import local_now, relative_time


st.set_page_config(page_title="이야기 공방", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

_SESSION_COOKIE = "novel_jepa_session"
CREATIVITY_LABELS = {"안정": "stable", "균형": "balanced", "대담": "bold"}
STATUS_LABELS = {
    JOB_QUEUED: "대기 중",
    JOB_RUNNING: "집필 중",
    JOB_SUCCEEDED: "완료",
    JOB_FAILED: "실패",
    JOB_FAILED_RECOVERABLE: "복구 가능",
}
SEARCH_THRESHOLD = 5  # 목록이 이만큼 쌓이면 검색창을 보인다
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
# 직접 작성 양식: (항목, 라벨, 위젯, 도움말). 글자 한도는 template_store.FIELD_LIMITS 하나를 쓴다.
MANUAL_FIELDS = (
    ("title", "제목", "input", ""),
    ("genre", "장르", "input", ""),
    ("premise", "핵심 소재", "area", "두세 문장. 무엇이 걸려 있고 무엇이 바뀌는가."),
    ("world", "세계관", "area", "시대·장소·규칙. 매 장 프롬프트에 그대로 들어가."),
    ("protagonist", "주인공", "area", "'이름: 한 줄 소개' 형태."),
    ("characters", "주요 인물 (선택)", "area", "'이름: 역할과 목표' 를 줄바꿈으로. 여기 없는 이름은 모델이 만들지 않아."),
    ("style_guide", "집필 지침 (선택)", "area", "문체·시점·금기·분위기. 예: 1인칭, 짧은 문장, 욕설 금지, 매 장 끝에 여운."),
)
REQUIRED_MANUAL = ("title", "genre", "premise", "world", "protagonist")
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
            --jade-dark: #07594f; --coral: #c65c4b; --paper: #f7f8f5; --white: #ffffff; --tint: #e4ece7;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        /* 헤더와 툴바는 남긴다: 접힌 사이드바를 펴는 » 버튼(stExpandSidebarButton)이 툴바 안에 있다.
           좁은 창(768px 이하)에서는 본문을 누르면 사이드바가 접히므로 이 버튼이 없으면 다시 열 수 없다.
           숨기는 건 배포 버튼·⋮ 메뉴(stToolbarActions)뿐. */
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
        [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
        [data-testid="stStatusWidget"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
        [data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] * { color: var(--jade) !important; }
        .block-container { max-width: 1100px; padding-top: 1.6rem; padding-bottom: 5rem; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        .stApp h1 { font-size: 1.8rem; line-height: 1.25; margin-bottom: 0.3rem; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important; border-radius: 10px !important; background: var(--white);
        }
        .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button,
        [data-testid="stPopoverButton"] { border-radius: 8px !important; min-height: 2.4rem; }
        /* 본문의 팝오버(⋯ · 🗑 · ✏️)는 메뉴 화살표 없이 아이콘만 */
        .stMain [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] { display: none; }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--jade) !important; border-color: var(--jade) !important; color: #fff !important;
        }
        [data-testid="stChatMessage"] { border-bottom: 1px solid var(--line); border-radius: 0; padding: 0.9rem 0.2rem; }
        /* 메시지 옆 도구(고치기·지우기·다시 생성)는 평소엔 옅게, 마우스를 올리면 또렷하게 */
        [data-testid="stChatMessage"] .stButton button[kind="tertiary"],
        [data-testid="stChatMessage"] [data-testid="stPopoverButton"] {
            color: var(--muted); min-height: 1.9rem; padding: 0 0.45rem; opacity: 0.45; border: none;
            background: transparent;
        }
        [data-testid="stChatMessage"]:hover .stButton button[kind="tertiary"],
        [data-testid="stChatMessage"]:hover [data-testid="stPopoverButton"] { opacity: 1; }
        [data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; }
        /* 사이드바 목록: 한 줄 말줄임, 촘촘한 간격, 열려 있는 항목만 옅은 배경 */
        .st-key-sidebar_list .stButton button { min-height: 2.1rem; padding: 0.2rem 0.6rem; }
        .st-key-sidebar_list .stButton button [data-testid="stMarkdownContainer"] { min-width: 0; overflow: hidden; }
        .st-key-sidebar_list .stButton button [data-testid="stMarkdownContainer"] p {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.92rem;
        }
        .st-key-sidebar_list .stButton button[kind="tertiary"] { color: var(--ink); }
        .st-key-sidebar_list .stButton button[kind="tertiary"]:hover { background: var(--tint); color: var(--ink); }
        .st-key-sidebar_list .stButton button[kind="secondary"] {
            background: var(--tint); border-color: transparent; color: var(--jade-dark); font-weight: 650;
        }
        .list-group { color: var(--muted); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
                      margin: 0.7rem 0 0.1rem 0.4rem; }
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
        .discarded-draft { color: var(--muted); white-space: pre-wrap; font-size: 0.92rem; line-height: 1.6; }
        .auto-note { border-left: 3px solid var(--jade); background: #eef3ef; padding: 0.5rem 0.8rem; font-size: 0.9rem; }
        .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.2rem; }
        .card-item b { color: var(--jade-dark); font-size: 0.8rem; letter-spacing: 0.04em; }
        .card-item div { white-space: pre-wrap; font-size: 0.93rem; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 0.7rem 5rem; }
            .card-grid { grid-template-columns: 1fr; }
            [data-testid="stChatMessage"] .stButton button[kind="tertiary"],
            [data-testid="stChatMessage"] [data-testid="stPopoverButton"] { opacity: 0.8; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_MARKDOWN_SPECIALS = re.compile(r"([\\`*_~\[\]<>#|$])")


def _md_label(text: str) -> str:
    """버튼 라벨은 마크다운으로 그려지므로 사용자가 적은 제목의 기호를 글자 그대로 보이게 한다."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", " ".join(str(text).split())) or "(제목 없음)"


def _file_stem(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\r\n]+', "_", str(text)).strip() or "원고"


# ---- 런타임 / 세션 ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_runtime() -> tuple[AppConfig, ConsumerStore, StoryTemplateStore, Any]:
    """설정·저장소·클라이언트는 프로세스에 하나면 된다 (저장소는 호출마다 연결을 새로 연다)."""
    config = load_config("configs/default.yaml")
    return config, ConsumerStore(config), StoryTemplateStore(config), make_llm_client(config)


def _model_status(client: Any) -> dict[str, Any]:
    cache_key = "consumer_model_status"
    cached = st.session_state.get(cache_key)
    now = time.time()
    if cached and now - cached["at"] < 5:
        return cached["status"]
    status = client.status()
    st.session_state[cache_key] = {"at": now, "status": status}
    return status


def _flash(message: str) -> None:
    """다음 실행에서 토스트로 보여줄 한 줄. st.rerun() 직전에 st.toast 를 부르면 사라지므로 세션에 맡긴다."""
    st.session_state["consumer_flash"] = message


def _show_flash() -> None:
    message = str(st.session_state.pop("consumer_flash", "") or "")
    if message:
        st.toast(message)


def _clear_story_session() -> None:
    for key in ("consumer_story_id", "consumer_last_job_status", "delete_story_confirmation"):
        st.session_state.pop(key, None)
    _clear_editor_session()


def _clear_editor_session() -> None:
    """원고 수정 패널의 상태는 작품마다 따로. 다른 작품으로 옮길 때 열려 있던 장·제안이 따라오지 않게."""
    for key in tuple(st.session_state.keys()):
        if str(key).startswith(("consumer_edit_", "edit_")):
            st.session_state.pop(key, None)


def _clear_account_session() -> None:
    for key in tuple(st.session_state.keys()):
        if str(key).startswith(("consumer_", "plan_", "story_", "tpl_", "edit_")) or key == "delete_story_confirmation":
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
            st.markdown('<div class="auth-note">작품은 계정별로 분리 보관되며 30일 뒤 자동 정리돼.</div>',
                        unsafe_allow_html=True)
    with artwork:
        st.image("assets/consumer/writing_studio.png", width="stretch")
        st.markdown('<div class="auth-art-caption">말로 기획하고, 장 단위로 완성하는 장편 소설</div>', unsafe_allow_html=True)


def _logout(store: ConsumerStore) -> None:
    token = str(st.session_state.get("consumer_session_token", ""))
    if token:
        store.revoke_user_session(token)
    _clear_account_session()
    st.session_state["consumer_logout_pending"] = True
    st.rerun()


# ---- 상태 표시 / 사이드바 ----------------------------------------------------------------------

def _status_bar(client: Any) -> None:
    """본문 상단의 모델 상태 한 줄. 사이드바는 화면 폭에 따라 접히므로 여기에 둔다."""
    status = _model_status(client)
    if status["ready"]:
        health = status.get("health") or {}
        vram = health.get("vram_gib")
        extra = f" · VRAM {vram} GiB" if vram is not None else ""
        st.caption(f"🟢 모델 준비됨 · {health.get('model', 'local')}{extra}")
    else:
        reason = " ".join(str(status.get("reason") or "").split())[:200]
        st.caption(f"🔴 모델을 쓸 수 없어 — {reason or 'run_service.bat 을 실행해줘.'}")


def _sidebar(store: ConsumerStore, user: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown('<div class="story-kicker">STORY STUDIO</div>', unsafe_allow_html=True)
        _sidebar_stories(store, user)
        st.divider()
        st.caption("로그인 계정")
        st.markdown(f"**{user['display_name']}**  \n@{user['username']}")
        if st.button("로그아웃", width="stretch", key="consumer_logout"):
            _logout(store)


def _sidebar_stories(store: ConsumerStore, user: dict[str, Any]) -> None:
    if st.button("➕ 새 작품 기획", width="stretch", key="story_new", type="primary"):
        _clear_story_session()
        st.session_state["plan_messages"] = []
        st.session_state["plan_card"] = {}
        st.rerun()
    stories = store.list_owned_stories(str(user["id"]))
    query = ""
    if len(stories) >= SEARCH_THRESHOLD:
        query = str(st.text_input("작품 검색", key="story_search", placeholder="🔍 제목이나 장르로 찾기",
                                  label_visibility="collapsed") or "").strip().casefold()
        if query:
            stories = [s for s in stories if query in str(s["title"]).casefold() or query in str(s["genre"]).casefold()]
    active = str(st.session_state.get("consumer_story_id", ""))
    now = local_now()
    with st.container(key="sidebar_list", gap=None):
        if not stories:
            st.caption("찾는 작품이 없어." if query else "아직 작품이 없어. 새 작품 기획에서 편하게 말해봐.")
        for story in stories:
            completed = bool(story.get("completed_at"))
            progress = 100.0 if completed else min(100.0, int(story["current_chars"]) / max(1, int(story["target_chars"])) * 100)
            state = "완결" if completed else f"진행 {progress:.0f}%"
            label = ("✓ " if completed else "") + _md_label(story["title"])
            meta = (f"{story['title']}\n\n{story['genre']} · {state} · {int(story['current_chars']):,}자 · "
                    f"{relative_time(story.get('updated_at'), now=now)}")
            if st.button(label, key=f"story_open_{story['id']}", width="stretch",
                         type="secondary" if story["id"] == active else "tertiary", help=meta):
                if story["id"] != active:
                    _clear_editor_session()
                st.session_state["consumer_story_id"] = story["id"]
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


def _start_story(config: AppConfig, store: ConsumerStore, user: dict[str, Any], card: dict[str, Any],
                 *, target: int, consent: bool, first_wish: str = "") -> None:
    """작품을 만들고 첫 턴을 큐에 넣은 뒤 집필 화면으로. 대화 기획과 직접 작성이 같이 쓴다."""
    story = store.create_story(
        str(user["id"]),
        title=str(card.get("title", "")), genre=str(card.get("genre", "")),
        premise=str(card.get("premise", "")), world=str(card.get("world", "")),
        protagonist=str(card.get("protagonist", "")), characters=str(card.get("characters", "") or ""),
        target_chars=int(target), research_consent=bool(consent), style_guide=str(card.get("style_guide", "") or ""),
    )
    instruction = "첫 장면을 시작해 줘." + (f" 참고: {first_wish[:300]}" if first_wish else "")
    try:
        store.enqueue_job(str(user["id"]), story["id"], instruction=instruction,
                          creativity_profile="balanced", requested_chars=config.consumer.default_turn_chars)
    except ConsumerStoreError as exc:
        # 작품은 만들어졌고 첫 턴만 못 넣은 것 (예: 점검 중). 집필 화면에서 다시 요청하면 된다.
        _flash(f"'{story['title']}' 을 만들었지만 첫 장 요청은 넣지 못했어: {exc}")
    else:
        _flash(f"'{story['title']}' 의 첫 장을 큐에 넣었어.")
    for key in ("plan_messages", "plan_card"):
        st.session_state.pop(key, None)
    for field, *_rest in MANUAL_FIELDS:
        st.session_state.pop(f"tpl_{field}", None)
    st.session_state["consumer_story_id"] = story["id"]
    st.rerun()


def _plan_chat(config: AppConfig, store: ConsumerStore, client: Any, user: dict[str, Any]) -> None:
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
            st.caption("세부를 손으로 다듬고 싶으면 옆 **📝 직접 작성** 탭에서 '작품 카드 가져오기' 를 눌러.")
            if st.button("✅ 이 설정으로 집필 시작", type="primary", disabled=not ready, key="plan_start", width="stretch"):
                first_wish = next((m["content"] for m in messages if m["role"] == "user"), "")
                try:
                    _start_story(config, store, user, card, target=int(target), consent=bool(consent), first_wish=first_wish)
                except (ValueError, ConsumerStoreError) as exc:
                    st.error(str(exc))

    status = _model_status(client)
    blocked = "" if status["ready"] else f"모델을 쓸 수 없어. {status['reason']}"
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


def _fill_manual_form(config: AppConfig, source: dict[str, Any]) -> None:
    """템플릿이나 작품 카드의 값을 직접 작성 양식에 채운다 (양식 위젯이 그려지기 전에 부른다)."""
    for field, *_rest in MANUAL_FIELDS:
        st.session_state[f"tpl_{field}"] = str(source.get(field, "") or "")
    st.session_state["tpl_target"] = _normalize_target(config, source.get("target_chars"))


def _manual_plan(config: AppConfig, store: ConsumerStore, templates: StoryTemplateStore, user: dict[str, Any]) -> None:
    user_id = str(user["id"])
    rows = templates.list_templates(user_id)
    if rows:
        by_name = {str(row["name"]): row for row in rows}
        pick_column, load_column, delete_column = st.columns([3, 1, 0.5], vertical_alignment="bottom")
        choice = pick_column.selectbox("내 템플릿", options=list(by_name), key="tpl_pick",
                                       help="저장해 둔 작품 설정. 불러오면 아래 양식에 채워져.")
        if load_column.button("불러오기", key="tpl_load", width="stretch"):
            _fill_manual_form(config, by_name[str(choice)])
            _flash(f"템플릿 '{choice}' 을 양식에 채웠어.")
            st.rerun()
        with delete_column.popover("🗑", help="이 템플릿 삭제"):
            st.caption(f"템플릿 '{choice}' 을 지워. 작품에는 영향 없어.")
            if st.button("정말 삭제", key="tpl_delete", type="primary", width="stretch"):
                try:
                    templates.delete_template(user_id, str(by_name[str(choice)]["id"]))
                except TemplateStoreError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("tpl_pick", None)
                    _flash(f"템플릿 '{choice}' 을 지웠어.")
                    st.rerun()
    else:
        st.caption("아직 저장한 템플릿이 없어. 양식을 채우고 **템플릿으로 저장** 을 누르면 다음 작품에서 불러올 수 있어.")
    card = st.session_state.get("plan_card") or {}
    if card and st.button("💬 대화로 만든 작품 카드 가져오기", key="tpl_import_card", help="대화 탭에서 정리한 카드를 양식에 채워."):
        _fill_manual_form(config, card)
        st.rerun()

    with st.form("manual_card_form"):
        values: dict[str, str] = {}
        title_column, genre_column = st.columns([1.2, 1])
        for field, label, widget, help_text in MANUAL_FIELDS:
            target_column = title_column if field == "title" else genre_column if field == "genre" else st
            limit = FIELD_LIMITS[field]
            if widget == "input":
                values[field] = target_column.text_input(label, key=f"tpl_{field}", max_chars=limit, help=help_text or None)
            else:
                values[field] = target_column.text_area(label, key=f"tpl_{field}", max_chars=limit, help=help_text or None,
                                                        height=120 if field in ("premise", "protagonist") else 150)
        options = config.consumer.target_char_options()
        target = st.selectbox("목표 분량", options=options, key="tpl_target", format_func=lambda value: f"{value:,}자")
        consent = st.checkbox("내 원고와 익명 품질 지표를 연구 개선에 활용하는 데 동의해", key="tpl_consent")
        name = st.text_input("템플릿 이름 (저장할 때만)", key="tpl_name", max_chars=TEMPLATE_NAME_CHARS,
                             placeholder="예: 항구 도시 미스터리 세계관")
        start_column, save_column = st.columns(2)
        start = start_column.form_submit_button("✅ 이 설정으로 집필 시작", type="primary", width="stretch")
        save = save_column.form_submit_button("💾 템플릿으로 저장", width="stretch")
    if save:
        try:
            saved = templates.save_template(user_id, name=name, target_chars=int(target),
                                            **{field: values[field] for field in TEMPLATE_FIELDS})
        except TemplateStoreError as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("tpl_pick", None)
            _flash(f"템플릿 '{saved['name']}' 을 저장했어.")
            st.rerun()
    if start:
        missing = [label for field, label, *_rest in MANUAL_FIELDS if field in REQUIRED_MANUAL and not values[field].strip()]
        if missing:
            st.error("비어 있는 항목: " + ", ".join(missing))
            return
        try:
            _start_story(config, store, user, values, target=int(target), consent=bool(consent))
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))


def _plan_mode(config: AppConfig, store: ConsumerStore, templates: StoryTemplateStore, client: Any, user: dict[str, Any]) -> None:
    st.markdown('<div class="story-kicker">NEW STORY</div>', unsafe_allow_html=True)
    st.title("어떤 이야기를 쓰고 싶어?")
    st.markdown('<div class="story-meta">말로 해도 되고, 양식을 직접 채워도 돼. 빈 곳은 대화에서 내가 제안해서 채울게.</div>',
                unsafe_allow_html=True)
    chat_tab, manual_tab = st.tabs(["💬 대화로 기획", "📝 직접 작성"])
    with chat_tab:
        _plan_chat(config, store, client, user)
    with manual_tab:
        _manual_plan(config, store, templates, user)


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


def _job_retry_notes(job: dict[str, Any]) -> list[str]:
    """끝난 턴에서 게이트가 한 일 — 워커가 metrics_json.retry_notes 에 남긴다."""
    try:
        metrics = json.loads(str(job.get("metrics_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return []
    notes = metrics.get("retry_notes") if isinstance(metrics, dict) else None
    return [str(item) for item in notes][:12] if isinstance(notes, list) else []


def _render_live_note(note: dict[str, Any] | None) -> None:
    """실시간 본문이 왜 사라졌는지: 게이트 사유, 버려진 초안, 걷어낸 문장 수, 채택 결정."""
    if not note:
        return
    kind = str(note.get("kind") or "")
    text = str(note.get("text") or "").strip()
    discarded = str(note.get("discarded") or "").strip()
    if kind == "retry":
        st.warning(f"🔁 {text} — 이 장을 다시 쓰는 중이야. 아래 초안은 저장되지 않고, 다시 쓸 때 참고하지도 않아.")
    elif kind == "repair":
        st.warning(f"🧭 {text} — 검토에서 나온 문제를 고쳐 다시 쓰는 중이야. 고치기 전 초안은 저장되지 않아.")
    elif kind == "trim":
        st.caption(f"✂️ {text}")
    elif kind == "decision":
        st.caption(("🧭 " if "개연성" in text else "🔁 ") + text)
    if discarded:
        label = "고치기 전 초안 보기 (저장되지 않음)" if kind == "repair" else "버려진 초안 보기 (저장되지 않음)"
        with st.expander(label):
            st.markdown(f'<div class="discarded-draft">{html.escape(discarded)}</div>', unsafe_allow_html=True)


def _render_job(job: dict[str, Any], sections: list[str], store: ConsumerStore, user_id: str,
                story_id: str = "", live_prose: str = "", note: dict[str, Any] | None = None) -> None:
    status = str(job["status"])
    job_id = int(job["id"])
    with st.chat_message("user"):
        instruction_column, delete_column = st.columns([9, 1], vertical_alignment="top")
        auto_turn = str(job.get("origin") or "") == JOB_ORIGIN_AUTO
        instruction = str(job["instruction"])
        if auto_turn and not instruction.startswith("🤖"):
            instruction = "🤖 " + instruction  # AI 가 정한 전개 — 사람이 적은 턴과 구분
        instruction_column.markdown(instruction)
        if story_id and status not in (JOB_QUEUED, JOB_RUNNING):
            if delete_column.button("✕", key=f"delete_job_{job_id}", type="tertiary", help="이 턴과 이 턴이 쓴 원고를 함께 지워."):
                try:
                    store.delete_owned_job(user_id, story_id, job_id)
                except ConsumerStoreError as exc:
                    st.error(str(exc))
                else:
                    _rerun_live()
    with st.chat_message("assistant"):
        if status == JOB_SUCCEEDED:
            st.session_state.pop(f"consumer_typed_{job_id}", None)
            start = int(job["start_section_count"])
            count = int(job["result_section_count"])
            generated = sections[start : start + count]
            if generated:
                for offset, section in enumerate(generated):
                    st.markdown(section)
                    section_index = start + offset + 1
                    if story_id and st.button("✏️ 이 장 수정", key=f"edit_section_{section_index}", type="tertiary",
                                              help="위의 '원고 수정' 패널에서 이 장을 연다."):
                        st.session_state["consumer_edit_target"] = section_index
                        _flash(f"{section_index}장을 위의 '원고 수정' 패널에서 열었어.")
                        st.rerun()
            else:
                st.caption("이 턴의 원고는 전체 원고 파일에 저장되어 있어.")
            # 게이트가 한 일: ✂️ 걷어냄 · 🔁 다시 씀 · 🧭 개연성 검토(점수·모순·고쳐 쓰기)
            for line in _job_retry_notes(job):
                st.caption(("✂️ " if "걷어냄" in line else "🧭 " if "개연성" in line else "🔁 ") + line)
        elif status == JOB_QUEUED:
            position = store.queue_position(user_id, job_id)
            st.markdown(f'<div class="queue-note">대기 순번 {position or "-"}번 · 요청을 안전하게 보관했어.</div>',
                        unsafe_allow_html=True)
        elif status == JOB_RUNNING:
            start = int(job["start_section_count"])
            committed = sections[start:]
            for section in committed:
                st.markdown(section)
            _render_live_note(note)
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
    for label, key in (("핵심 소재", "premise"), ("세계관", "world"), ("주인공", "protagonist"), ("주요 인물", "characters"),
                       ("집필 지침", "style_guide")):
        value = str(story.get(key) or "").strip()
        st.markdown(f"**{label}**")
        st.markdown(f'<div class="settings-block">{html.escape(value) or "(비어 있음)"}</div>', unsafe_allow_html=True)
    st.caption("연구 활용 동의: " + ("동의함" if int(story["research_consent"] or 0) else "동의하지 않음"))


def _settings_form(config: AppConfig, store: ConsumerStore, user_id: str, story: dict[str, Any]) -> None:
    story_id = str(story["id"])
    with st.form(f"edit_story_{story_id}"):
        title_column, genre_column = st.columns([1.2, 1])
        title = title_column.text_input("제목", value=str(story["title"]), max_chars=FIELD_LIMITS["title"])
        genre = genre_column.text_input("장르", value=str(story["genre"]), max_chars=FIELD_LIMITS["genre"])
        premise = st.text_area("핵심 소재", value=str(story["premise"]), height=100, max_chars=FIELD_LIMITS["premise"])
        world = st.text_area("세계관", value=str(story["world"]), height=140, max_chars=FIELD_LIMITS["world"])
        protagonist_column, characters_column = st.columns(2)
        protagonist = protagonist_column.text_area("주인공", value=str(story["protagonist"]), height=120,
                                                   max_chars=FIELD_LIMITS["protagonist"])
        characters = characters_column.text_area("주요 인물 (선택)", value=str(story["characters"] or ""), height=120,
                                                 max_chars=FIELD_LIMITS["characters"])
        style_guide = st.text_area("집필 지침 (선택)", value=str(story.get("style_guide") or ""), height=100,
                                   max_chars=FIELD_LIMITS["style_guide"],
                                   help="문체·시점·금기·분위기. 매 장 프롬프트에 규칙으로 들어가. 예: 1인칭, 짧은 문장, 욕설 금지.")
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
                                     research_consent=consent, style_guide=style_guide)
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))
        else:
            st.success("설정을 저장했어. 다음 턴부터 반영돼.")
            st.rerun()


def _rerun_live() -> None:
    """집필 화면(fragment) 안에서의 새로고침. 브라우저에서는 fragment 만 다시 돌고, AppTest 처럼 전체 실행 중이면
    scope="fragment" 가 거부되므로 전체 새로고침으로 물러난다."""
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _enqueue_auto_turn(store: ConsumerStore, user_id: str, story_id: str, creativity: str, turn_chars: int) -> bool:
    """AI 가 전개를 정하는 턴 하나를 큐에 넣는다. 지시는 워커가 집을 때 채운다."""
    try:
        store.enqueue_job(user_id, story_id, instruction=AUTO_PLACEHOLDER, creativity_profile=creativity,
                          requested_chars=turn_chars, origin=JOB_ORIGIN_AUTO)
    except (ValueError, ConsumerStoreError) as exc:
        st.error(str(exc))
        return False
    return True


def _auto_controls(config: AppConfig, store: ConsumerStore, user_id: str, story: dict[str, Any], *,
                   outstanding: dict[str, Any] | None, completed: bool, blocked: str, creativity: str,
                   turn_chars: int) -> None:
    """자동 이어쓰기: 켜 두면 워커가 턴이 끝날 때마다 AI 가 정한 전개로 다음 턴을 넣는다 (완결까지)."""
    story_id = str(story["id"])
    auto_on = bool(story.get("auto_continue"))
    limit = int(config.consumer.auto_continue_max_turns)
    left, right = st.columns([1.4, 1], vertical_alignment="center")
    if auto_on:
        left.markdown(
            '<div class="auto-note">🤖 <b>자동 이어쓰기 켜짐</b> — 턴이 끝나면 AI 가 다음 전개를 정해 바로 이어 써. '
            f'완결되거나 연속 {limit}턴이 되면 멈춰. 직접 전개를 적어 끼어들어도 돼.</div>',
            unsafe_allow_html=True,
        )
        if right.button("자동 이어쓰기 끄기", key="auto_off", width="stretch",
                        help="지금 쓰는 턴은 끝까지 쓰고 거기서 멈춰."):
            try:
                store.set_auto_continue(user_id, story_id, False)
            except ConsumerStoreError as exc:
                st.error(str(exc))
            else:
                _rerun_live()
        return
    if left.button("🤖 자동 이어쓰기 켜기", key="auto_on", width="stretch", disabled=completed or store.is_maintenance(),
                   help="사람이 다음 전개를 적지 않아도 AI 가 정해서 완결까지 이어 써. 언제든 끌 수 있어."):
        try:
            store.set_auto_continue(user_id, story_id, True)
        except ConsumerStoreError as exc:
            st.error(str(exc))
        else:
            # 기다리는 턴이 없으면 바로 첫 자동 턴을 넣는다 — 켰는데 아무 일도 안 일어나면 안 되니까.
            if outstanding is None:
                _enqueue_auto_turn(store, user_id, story_id, creativity, turn_chars)
            _rerun_live()
    if right.button("🤖 다음 한 턴만 AI 에게", key="auto_once", width="stretch", disabled=bool(blocked),
                    help="이번 턴의 전개만 AI 가 정한다. 그 뒤는 다시 네가."):
        if _enqueue_auto_turn(store, user_id, story_id, creativity, turn_chars):
            _rerun_live()


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
    live_note = read_live_note(workspace)
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
                    _rerun_live()
        for job in jobs:
            _render_job(job, sections, store, user_id, story_id=story_id, live_prose=live_prose, note=live_note)
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

    _auto_controls(config, store, user_id, story, outstanding=outstanding, completed=completed,
                   blocked=blocked_reason, creativity=CREATIVITY_LABELS.get(str(creativity_label), "balanced"),
                   turn_chars=int(turn_chars))

    prompt = st.chat_input("다음 전개를 말해줘 (예: 박 노인이 20년 전 일을 털어놓게 해줘)",
                           disabled=bool(blocked_reason), key="consumer_chat_input")
    if prompt:
        try:
            store.enqueue_job(user_id, story_id, instruction=prompt,
                              creativity_profile=CREATIVITY_LABELS.get(str(creativity_label), "balanced"),
                              requested_chars=int(turn_chars))
            _rerun_live()
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))

    if queue["queued"] or queue["running"]:
        st.caption(f"전체 대기 {queue['queued']}건 · 실행 {queue['running']}건")

    st.divider()
    download_a, download_b = st.columns(2)
    # 내려받기 내용은 눌렀을 때만 만든다 — 이 fragment 는 1초마다 다시 도니까 매번 zip 을 만들면 낭비다.
    download_a.download_button("전체 원고", data=lambda: draft.encode("utf-8"), file_name=f"{_file_stem(story['title'])}.md",
                               mime="text/markdown", disabled=not bool(draft), width="stretch")
    download_b.download_button("이어쓰기 번들", data=lambda: build_continuation_bundle(workspace, story),
                               file_name=f"{_file_stem(story['title'])}_continue.zip", mime="application/zip", width="stretch")

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
                    _rerun_live()
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


def _apply_section_edit(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story_id: str,
                        index: int, text: str, refresh: bool) -> None:
    try:
        result = replace_section(config, store, client, user_id, story_id, index, text, refresh_memory=refresh)
    except (StoryEditError, ConsumerStoreError, ValueError, LocalLLMError) as exc:
        st.error(str(exc))
        return
    summary = str(result["memory"].summary)[:60]
    if result["changed"]:
        _flash(f"{index}장을 저장했어." + (f" 요약: {summary}" if refresh else ""))
    elif refresh:
        _flash(f"본문은 그대로, {index}장 요약 메모리만 다시 만들었어: {summary}")
    else:
        st.info("바뀐 내용이 없어.")
        return
    _forget_editor_widgets(story_id)
    st.session_state["consumer_edit_target"] = index
    st.session_state["consumer_edit_open"] = True
    st.rerun()


def _content_key(*parts: Any) -> str:
    """위젯 키에 내용 해시를 섞는다. 장 번호만 쓰면 턴 삭제로 번호가 밀리거나 메모리를 새로 뽑아도 옛 입력이 남아
    다른 장 위에 저장될 수 있다 (키 있는 text_area 는 value= 가 바뀌어도 상태를 유지한다)."""
    digest = hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return digest


def _editor_manual(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story_id: str,
                   current: Any, busy: bool) -> None:
    text = st.text_area("본문 (첫 줄은 '### 소제목')", value=current.text,
                        key=f"edit_text_{story_id}_{current.index}_{_content_key(current.text)}", height=380, disabled=busy)
    refresh = st.checkbox("저장할 때 이 장의 요약 메모리를 AI 로 다시 만든다", value=True, key=f"edit_refresh_{story_id}",
                          help="끄면 기존 메모리를 그대로 둬. 사건·사실이 바뀌었으면 켜 두는 게 안전해.")
    st.caption("저장하면 본문·요약 메모리·원장·진행 지표가 함께 바뀌고, 이어쓰기는 고친 내용 위에서 계속돼. "
               f"지금 {len(text):,}자.")
    if st.button("💾 저장하고 메모리 갱신" if refresh else "💾 저장", type="primary", key=f"edit_save_{story_id}",
                 disabled=busy, width="stretch"):
        _apply_section_edit(config, store, client, user_id, story_id, current.index, text, refresh)


def _editor_ai(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story: dict[str, Any],
               current: Any, busy: bool, model_ready: bool) -> None:
    story_id = str(story["id"])
    index = current.index
    stamp = _content_key(current.text)
    draft_key = f"edit_ai_draft_{story_id}_{index}_{stamp}"
    text_key = f"edit_ai_text_{story_id}_{index}_{_content_key(current.text, st.session_state.get(draft_key, ''))}"
    quick = st.pills("빠른 요청", QUICK_REWRITES, selection_mode="single", key=f"edit_quick_{story_id}_{index}", disabled=busy)
    custom = st.text_input("직접 요청", placeholder="예: 박 노인의 대사를 더 거칠게. 마지막 문장은 그대로.",
                           key=f"edit_custom_{story_id}_{index}", disabled=busy)
    instruction = " / ".join(part for part in (str(quick or ""), str(custom or "").strip()) if part)
    if not model_ready:
        st.caption("🔴 모델을 쓸 수 없어 AI 보조가 잠겼어.")
    if st.button("🤖 AI 에게 맡기기", key=f"edit_ai_go_{story_id}_{index}", width="stretch",
                 disabled=busy or not instruction or not model_ready,
                 help="원문은 바꾸지 않고 제안만 먼저 보여줘. 마음에 들면 저장해."):
        with st.container(border=True):
            st.caption(f"요청: {instruction}")
            try:
                with st.spinner("고칠 곳을 고르는 중..."):
                    plan = plan_edits(client, config, story, current.text, instruction)
                if plan:
                    st.markdown("**수정 계획**  " + chr(10) + "  ".join(
                        f"- '{edit['where']}' → {edit['how']}" + chr(10) for edit in plan))
                streamed = st.write_stream(stream_rewrite(client, config, story, current.text, instruction, plan))
                text = streamed if isinstance(streamed, str) else "".join(str(part) for part in streamed)
                # 4B 모델은 같은 요청도 어떤 장에선 반토막을 낸다. 범위를 벗어나면 이유를 붙여 한 번 더 쓴다.
                note = retry_note_for(current.text, text, instruction)
                if note:
                    st.caption(f"제안이 원문의 {len(text) / max(1, len(current.text)) * 100:.0f}% 라서 한 번 더 쓰는 중…")
                    retried = st.write_stream(stream_rewrite(client, config, story, current.text, instruction, plan, note))
                    retried_text = retried if isinstance(retried, str) else "".join(str(part) for part in retried)
                    if abs(len(retried_text) / max(1, len(current.text)) - 1) < abs(len(text) / max(1, len(current.text)) - 1):
                        text = retried_text
            except LocalLLMError as exc:
                st.error(str(exc))
                return
        try:
            st.session_state[draft_key] = normalize_edit(text, index, current.title, from_model=True)
        except StoryEditError as exc:
            st.error(str(exc))
            return
        st.session_state.pop(text_key, None)
        st.rerun()
    draft = st.session_state.get(draft_key)
    if not draft:
        st.caption("제안은 원문 아래에 따로 나와. 고쳐서 저장하거나 버릴 수 있어.")
        return
    # text_area 를 캡션보다 먼저: 제안이 사라진 다음 실행에서 같은 자리에 안내 캡션이 들어가 AppTest 의 낡은
    # 노드를 덮는다 (실제 프런트는 낡은 노드를 알아서 지운다).
    edited = st.text_area("AI 제안 — 더 고쳐서 저장할 수 있어", value=str(draft), key=text_key, height=380)
    ratio = len(str(draft)) / max(1, len(current.text))
    low, high = length_bounds(instruction)
    note = "" if low <= ratio <= high else " · 분량이 많이 달라졌어. 저장 전에 빠진 내용이 없는지 봐줘."
    st.caption(f"원문 {len(current.text):,}자 → 제안 {len(str(draft)):,}자 ({ratio * 100:.0f}%){note}")
    save_column, drop_column = st.columns(2)
    if save_column.button("💾 이 내용으로 저장", type="primary", key=f"edit_ai_save_{story_id}_{index}", disabled=busy, width="stretch"):
        _apply_section_edit(config, store, client, user_id, story_id, index, edited, True)
    if drop_column.button("버리기", key=f"edit_ai_drop_{story_id}_{index}", width="stretch"):
        st.session_state.pop(draft_key, None)
        st.session_state.pop(text_key, None)
        st.rerun()


def _editor_memory(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story_id: str,
                   workspace: StoryWorkspace, index: int, busy: bool, model_ready: bool) -> None:
    memory = section_memory(workspace, index)
    if memory is None:
        st.caption("이 장의 메모리가 아직 없어. 'AI 로 다시 추출' 을 누르면 만들어져.")
    else:
        st.caption("다음 장을 쓸 때 모델은 본문 대신 이 메모리를 본다. 틀린 사실이 있으면 여기서 바로 고쳐.")
    # 메모리 내용으로 키를 만든다: 저장·재추출로 메모리가 바뀌면 입력칸도 새 값으로 바뀐다.
    stamp = _content_key(memory.model_dump_json() if memory else "")
    summary = st.text_area("한 줄 요약", value=memory.summary if memory else "",
                           key=f"edit_mem_summary_{story_id}_{index}_{stamp}", height=80, disabled=busy)
    facts = st.text_area("확정된 사실 (줄마다 하나)", value="\n".join(memory.facts) if memory else "",
                         key=f"edit_mem_facts_{story_id}_{index}_{stamp}", height=120, disabled=busy)
    clues = st.text_area("미해결 단서 (줄마다 하나)", value="\n".join(memory.open_clues) if memory else "",
                         key=f"edit_mem_clues_{story_id}_{index}_{stamp}", height=100, disabled=busy)
    save_column, refresh_column = st.columns(2)
    if save_column.button("💾 메모리 저장", type="primary", key=f"edit_mem_save_{story_id}_{index}", disabled=busy, width="stretch"):
        try:
            saved = update_section_memory(config, store, user_id, story_id, index, summary=summary, facts=facts, open_clues=clues)
        except (StoryEditError, ConsumerStoreError) as exc:
            st.error(str(exc))
        else:
            _flash(f"{index}장 메모리를 저장했어: {saved.summary[:60]}")
            _forget_editor_widgets(story_id)
            st.rerun()
    if refresh_column.button("🤖 AI 로 다시 추출", key=f"edit_mem_refresh_{story_id}_{index}", width="stretch",
                             disabled=busy or not model_ready, help="본문은 그대로 두고 요약·사실·단서를 모델이 다시 뽑아."):
        try:
            refreshed = refresh_section_memory(config, store, client, user_id, story_id, index)
        except (StoryEditError, ConsumerStoreError, LocalLLMError) as exc:
            st.error(str(exc))
        else:
            _flash(f"{index}장 요약을 다시 만들었어: {refreshed.summary[:60]}")
            _forget_editor_widgets(story_id)
            st.rerun()


def _forget_editor_widgets(story_id: str) -> None:
    """AI 제안(세션 값)을 치운다. 위젯 키(edit_text_/edit_ai_text_/edit_mem_)는 내용 해시가 바뀌면 저절로
    새 위젯이 되므로 건드리지 않는다 — 직전 실행의 위젯 키를 지우면 AppTest 가 그 위젯 상태를 못 찾는다."""
    for key in tuple(st.session_state.keys()):
        if str(key).startswith("edit_ai_draft_") and f"_{story_id}_" in str(key):
            st.session_state.pop(key, None)


def _section_editor(config: AppConfig, store: ConsumerStore, client: Any, user_id: str, story: dict[str, Any]) -> None:
    """장 단위 원고 수정 패널. 집필 화면의 1초 폴링 fragment 바깥에 둔다 (입력 중 깜빡임 방지)."""
    story_id = str(story["id"])
    workspace = StoryWorkspace.for_story(config, story_id)
    sections = list_sections(workspace)
    target = st.session_state.pop("consumer_edit_target", None)
    if target is not None:
        st.session_state["consumer_edit_open"] = True
    selected = int(target) if target is not None else int(st.session_state.get("consumer_edit_section") or 0)
    # 라벨에 고른 장을 넣는다: 라벨이 바뀌면 새 expander 가 되어 접어 둔 패널도 ✏️ 를 누르면 다시 펼쳐진다.
    label = "✏️ 원고 수정 · 직접 고치거나 AI 에게 맡기기" + (f" · {selected}장" if selected else "")
    with st.expander(label, expanded=bool(st.session_state.get("consumer_edit_open"))):
        if not sections:
            st.caption("아직 쓴 장이 없어. 첫 턴이 끝나면 여기서 장을 고칠 수 있어.")
            return
        options = [section.index for section in sections]
        # 라벨은 번호·소제목만: 글자 수처럼 저장할 때마다 바뀌는 값을 넣으면 브라우저가 기억한 선택지 문자열이
        # 어긋나 다음 상호작용에서 1장으로 되돌아간다.
        labels = {section.index: f"{section.index}장 · {section.title}" for section in sections}
        # 키를 pop 하고 index= 로 다시 만들면 백엔드만 바뀌고 브라우저 선택은 그대로다. 값을 직접 넣어야 동기화된다.
        if target in options:
            st.session_state["consumer_edit_section"] = int(target)
        elif st.session_state.get("consumer_edit_section") not in options:
            st.session_state["consumer_edit_section"] = options[0]
        index = int(st.selectbox("수정할 장", options=options, format_func=lambda value: labels[value],
                                 key="consumer_edit_section"))
        current = sections[index - 1]
        st.caption(f"{current.chars:,}자" + (" · 완결된 작품이야. 결말 장을 고쳐도 완결 상태는 유지돼." if story.get("completed_at") else ""))
        busy = store.owned_outstanding_job(user_id, story_id) is not None
        model_ready = bool(_model_status(client)["ready"])
        if busy:
            st.warning("집필 중이거나 대기 중인 요청이 있어. 끝난 뒤에 고칠 수 있어.")
        manual_tab, ai_tab, memory_tab = st.tabs(["직접 수정", "AI 보조", "요약 메모리"])
        with manual_tab:
            _editor_manual(config, store, client, user_id, story_id, current, busy)
        with ai_tab:
            _editor_ai(config, store, client, user_id, story, current, busy, model_ready)
        with memory_tab:
            _editor_memory(config, store, client, user_id, story_id, workspace, index, busy, model_ready)


def _story_mode(config: AppConfig, store: ConsumerStore, templates: StoryTemplateStore, client: Any,
                user: dict[str, Any], story: dict[str, Any]) -> None:
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
        view_tab, edit_tab, template_tab = st.tabs(["설정 보기", "설정 수정", "템플릿으로 저장"])
        with view_tab:
            _settings_view(story)
        with edit_tab:
            st.caption("세계관과 인물은 매 장 프롬프트에 다시 들어가. 수정하면 다음 턴부터 반영되고, 이미 쓴 본문은 그대로 남아.")
            _settings_form(config, store, str(user["id"]), story)
        with template_tab:
            st.caption("이 작품의 장르·소재·세계관·인물·집필 지침을 템플릿으로 저장해 두면 새 작품의 **📝 직접 작성** 에서 불러올 수 있어.")
            with st.form(f"story_template_{story['id']}", border=False):
                name = st.text_input("템플릿 이름", value=str(story["title"]), max_chars=TEMPLATE_NAME_CHARS)
                if st.form_submit_button("💾 템플릿으로 저장", width="stretch"):
                    try:
                        saved = templates.save_template(str(user["id"]), name=name, target_chars=int(story["target_chars"]),
                                                        **template_from_story(story))
                    except TemplateStoreError as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"템플릿 '{saved['name']}' 으로 저장했어.")
    _section_editor(config, store, client, str(user["id"]), story)
    _story_live(config, store, client, str(user["id"]), str(story["id"]))


# ---- 메인 --------------------------------------------------------------------------------------

def main() -> None:
    _styles()
    config, store, templates, client = _load_runtime()
    if st.session_state.pop("consumer_logout_pending", False):
        _forget_session_cookie()
    user = _current_user(store)
    if user is None:
        _auth_entry(config, store)
        return
    _sync_session_cookie(str(st.session_state.get("consumer_session_token", "")), config.consumer.auth_session_days)
    _show_flash()

    _status_bar(client)
    _sidebar(store, user)
    story_id = str(st.session_state.get("consumer_story_id", ""))
    story = store.get_owned_story(str(user["id"]), story_id) if story_id else None
    if story is None:
        _clear_story_session()
        _plan_mode(config, store, templates, client, user)
        return
    _story_mode(config, store, templates, client, user, story)


if __name__ == "__main__":
    main()
