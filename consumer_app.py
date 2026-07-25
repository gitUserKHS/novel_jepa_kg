from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from src.service.artifacts import active_model_status
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
from src.service.story_workspace import (
    StoryWorkspace,
    build_continuation_bundle,
    read_draft,
    split_sections,
)
from src.memory.story_outline import load_story_outline
from src.utils.config import AppConfig, load_config


st.set_page_config(page_title="이야기 공방", page_icon="✦", layout="wide")


CREATIVITY_LABELS = {
    "안정": "stable",
    "균형": "balanced",
    "대담": "bold",
}
STATUS_LABELS = {
    JOB_QUEUED: "대기 중",
    JOB_RUNNING: "집필 중",
    JOB_SUCCEEDED: "완료",
    JOB_FAILED: "실패",
    JOB_FAILED_RECOVERABLE: "복구 가능",
}


def _styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #202624;
            --muted: #68716d;
            --line: #d9dfdb;
            --jade: #0b7569;
            --jade-dark: #07594f;
            --coral: #c65c4b;
            --paper: #f7f8f5;
            --white: #ffffff;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { height: 0; background: transparent; }
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
            display: none !important;
        }
        .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        .stApp h1 { font-size: 2rem; line-height: 1.25; margin-bottom: 0.35rem; }
        .stApp h2 { font-size: 1.35rem; }
        .stApp p, .stApp label { letter-spacing: 0; }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 8px !important;
            background: var(--white);
        }
        .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
            border-radius: 6px !important;
            min-height: 2.65rem;
        }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--jade) !important;
            border-color: var(--jade) !important;
            color: #ffffff !important;
        }
        .stButton button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: var(--jade-dark) !important;
            border-color: var(--jade-dark) !important;
        }
        [data-baseweb="tab-list"] { gap: 1rem; }
        [data-baseweb="tab-highlight"] { background-color: var(--jade) !important; }
        [role="tab"][aria-selected="true"] { color: var(--jade-dark) !important; }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--jade) !important;
            box-shadow: 0 0 0 1px var(--jade) !important;
        }
        [data-testid="stMetric"] {
            background: transparent;
            border-left: 2px solid var(--line);
            padding: 0.25rem 0.9rem;
        }
        [data-testid="stChatMessage"] {
            border-bottom: 1px solid var(--line);
            border-radius: 0;
            padding: 1rem 0.2rem;
        }
        .story-kicker {
            color: var(--jade);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .story-meta { color: var(--muted); font-size: 0.9rem; }
        .auth-copy { color: var(--muted); font-size: 1rem; margin: 0 0 1.4rem; }
        .auth-note {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 1.2rem;
            padding-top: 1rem;
        }
        .auth-art-caption {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.6rem;
        }
        [data-testid="stHorizontalBlock"]:has(.auth-anchor) [data-testid="stColumn"]:last-child img {
            aspect-ratio: 4 / 3;
            border-radius: 8px;
            object-fit: cover;
        }
        .outline-strip {
            border-left: 3px solid var(--jade);
            background: #eef3ef;
            padding: 0.7rem 0.85rem;
            margin: 0.8rem 0 1.15rem;
        }
        .outline-label { color: var(--jade-dark); font-size: 0.76rem; font-weight: 750; }
        .outline-text { color: var(--ink); font-size: 0.93rem; margin-top: 0.15rem; }
        .queue-note {
            border-left: 4px solid var(--coral);
            padding: 0.3rem 0.8rem;
            color: var(--ink);
        }
        @media (max-width: 640px) {
            .block-container { padding: 1.3rem 0.8rem 5rem; }
            .stApp h1 { font-size: 1.72rem; line-height: 1.25; }
            [data-testid="stHorizontalBlock"] { gap: 0.45rem; }
            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
                width: 100% !important;
                min-width: 0 !important;
            }
            [data-testid="stMetric"] { min-height: 70px; padding: 0.35rem 0.55rem; }
            [data-testid="stMetricValue"] { font-size: 1.35rem; }
            [data-testid="stHorizontalBlock"]:has(.auth-anchor) {
                display: flex;
                flex-direction: column;
            }
            [data-testid="stHorizontalBlock"]:has(.auth-anchor) [data-testid="stColumn"] {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_runtime() -> tuple[AppConfig, ConsumerStore]:
    config = load_config("configs/default.yaml")
    config.ollama.chat_model = config.consumer.chat_model
    config.ollama.embed_model = config.consumer.embed_model
    return config, ConsumerStore(config)


def _clear_story_session() -> None:
    for key in (
        "consumer_story_id",
        "consumer_last_job_status",
        "delete_story_confirmation",
    ):
        st.session_state.pop(key, None)


def _clear_account_session() -> None:
    for key in tuple(st.session_state.keys()):
        if str(key).startswith("consumer_") or key == "delete_story_confirmation":
            st.session_state.pop(key, None)


def _current_user(store: ConsumerStore) -> dict[str, Any] | None:
    token = str(st.session_state.get("consumer_session_token", ""))
    if not token:
        return None
    user = store.authenticate_user_session(token)
    if user is None:
        _clear_account_session()
    return user


def _finish_login(store: ConsumerStore, user: dict[str, Any]) -> None:
    st.session_state["consumer_session_token"] = store.create_user_session(str(user["id"]))
    _clear_story_session()
    st.rerun()


def _auth_entry(config: AppConfig, store: ConsumerStore) -> None:
    auth, artwork = st.columns([0.86, 1.14], gap="large", vertical_alignment="center")
    with auth:
        st.markdown('<div class="auth-anchor"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="story-kicker">STORY STUDIO</div>', unsafe_allow_html=True)
            st.title(config.consumer.name)
            st.markdown(
                '<div class="auth-copy">한 문장으로 시작해, 한 권의 이야기로.</div>',
                unsafe_allow_html=True,
            )
            login_tab, register_tab = st.tabs(["로그인", "회원가입"])

            with login_tab:
                with st.form("consumer_login_form", clear_on_submit=False, border=False):
                    username = st.text_input("아이디", max_chars=40, autocomplete="username")
                    password = st.text_input(
                        "비밀번호",
                        type="password",
                        max_chars=128,
                        autocomplete="current-password",
                    )
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
                    username = st.text_input(
                        "새 아이디",
                        max_chars=40,
                        autocomplete="username",
                        help="영문 또는 숫자로 시작하는 3~40자의 영문, 숫자, ., _, -",
                    )
                    password = st.text_input(
                        "새 비밀번호",
                        type="password",
                        max_chars=128,
                        autocomplete="new-password",
                    )
                    password_confirm = st.text_input(
                        "비밀번호 확인",
                        type="password",
                        max_chars=128,
                        autocomplete="new-password",
                    )
                    submitted = st.form_submit_button("회원가입", type="primary", width="stretch")
                if submitted:
                    if password != password_confirm:
                        st.error("비밀번호 확인이 일치하지 않아.")
                    else:
                        try:
                            user = store.create_user(
                                username=username,
                                display_name=display_name,
                                password=password,
                            )
                            _finish_login(store, user)
                        except (ValueError, AccountExistsError) as exc:
                            st.error(str(exc))
            st.markdown(
                '<div class="auth-note">작품은 계정별로 분리 보관되며 30일 뒤 자동 정리돼.</div>',
                unsafe_allow_html=True,
            )
    with artwork:
        st.image("assets/consumer/writing_studio.png", width="stretch")
        st.markdown(
            '<div class="auth-art-caption">장면과 기억을 차곡차곡 엮는 장편 집필 공간</div>',
            unsafe_allow_html=True,
        )


def _logout(store: ConsumerStore) -> None:
    token = str(st.session_state.get("consumer_session_token", ""))
    if token:
        store.revoke_user_session(token)
    _clear_account_session()
    st.rerun()


def _account_sidebar(store: ConsumerStore, user: dict[str, Any]) -> None:
    with st.sidebar:
        st.caption("로그인 계정")
        st.subheader(str(user["display_name"]))
        st.caption(f"@{user['username']}")
        if st.button("로그아웃", width="stretch", key="consumer_logout"):
            _logout(store)


def _activate_story(story: dict[str, Any]) -> None:
    st.session_state["consumer_story_id"] = story["id"]


def _story_entry(config: AppConfig, store: ConsumerStore, user: dict[str, Any]) -> None:
    _account_sidebar(store, user)
    st.markdown('<div class="story-kicker">LIBRARY</div>', unsafe_allow_html=True)
    st.title("내 작품")
    st.markdown(
        '<div class="story-meta">쓰던 이야기를 이어가거나 새 세계를 열어봐.</div>',
        unsafe_allow_html=True,
    )
    library_tab, new_tab = st.tabs(["내 작품", "새 작품"])

    with library_tab:
        stories = store.list_owned_stories(str(user["id"]))
        if not stories:
            st.info("아직 만든 작품이 없어. 새 작품 탭에서 첫 이야기를 시작해봐.")
        for story in stories:
            with st.container(border=True):
                summary, action = st.columns([5, 1], vertical_alignment="center")
                with summary:
                    st.markdown(f"**{story['title']}**")
                    progress = min(
                        100.0,
                        int(story["current_chars"]) / max(1, int(story["target_chars"])) * 100,
                    )
                    completion = "완결 · " if progress >= 100.0 else ""
                    st.caption(
                        f"{completion}{story['genre']} · {int(story['current_chars']):,} / "
                        f"{int(story['target_chars']):,}자 · {progress:.1f}%"
                    )
                    st.progress(progress / 100.0)
                with action:
                    if st.button("열기", key=f"open_story_{story['id']}", width="stretch"):
                        _activate_story(story)
                        st.rerun()

    with new_tab:
        with st.form("new_story_form"):
            title_column, genre_column = st.columns([1.2, 1])
            with title_column:
                title = st.text_input("제목", max_chars=100)
            with genre_column:
                genre = st.text_input("장르", placeholder="예: SF 미스터리", max_chars=80)
            premise = st.text_area("핵심 소재", height=100, max_chars=1500)
            world = st.text_area("세계관", height=130, max_chars=4000)
            protagonist_column, characters_column = st.columns(2)
            with protagonist_column:
                protagonist = st.text_area(
                    "주인공",
                    height=110,
                    max_chars=2000,
                    placeholder="예: 서윤: 잃어버린 기억을 추적하는 기록 복원가",
                )
            with characters_column:
                characters = st.text_area(
                    "주요 인물 (선택)",
                    height=110,
                    max_chars=3000,
                    placeholder="한 줄에 한 명씩 이름: 역할과 목표",
                )
            target_char_options = config.consumer.target_char_options()
            target_chars = st.selectbox(
                "전체 목표 글자 수",
                options=target_char_options,
                index=target_char_options.index(config.consumer.default_target_chars),
                format_func=lambda value: f"{value:,}자",
            )
            consent = st.checkbox("내 원고와 익명 품질 지표를 연구 개선에 활용하는 데 동의해")
            submitted = st.form_submit_button("작품 만들기", type="primary", width="stretch")
        if submitted:
            try:
                story = store.create_story(
                    str(user["id"]),
                    title=title,
                    genre=genre,
                    premise=premise,
                    world=world,
                    protagonist=protagonist,
                    characters=characters,
                    target_chars=int(target_chars),
                    research_consent=consent,
                )
                _activate_story(story)
                st.rerun()
            except (ValueError, ConsumerStoreError) as exc:
                st.error(str(exc))


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
        status = str(payload.get("status", "unknown"))
        labels = {
            "idle": "준비됨",
            "polling": "준비됨",
            "running": "집필 중",
            "waiting_for_gpu": "자원 대기",
            "maintenance": "점검 중",
            "model_unavailable": "모델 준비 중",
        }
        return labels.get(status, status), fresh
    except (ValueError, TypeError, json.JSONDecodeError):
        return "상태 확인 필요", False


def _render_job(
    job: dict[str, Any],
    sections: list[str],
    store: ConsumerStore,
    user_id: str,
) -> None:
    with st.chat_message("user"):
        st.markdown(str(job["instruction"]))
    status = str(job["status"])
    with st.chat_message("assistant"):
        if status == JOB_SUCCEEDED:
            start = int(job["start_section_count"])
            count = int(job["result_section_count"])
            generated = sections[start : start + count]
            if generated:
                for section in generated:
                    st.markdown(section)
            else:
                st.caption("이 턴의 원고는 전체 원고 파일에 저장되어 있어.")
        elif status == JOB_QUEUED:
            position = store.queue_position(user_id, int(job["id"]))
            st.markdown(
                f'<div class="queue-note">대기 순번 {position or "-"}번 · 요청을 안전하게 보관했어.</div>',
                unsafe_allow_html=True,
            )
        elif status == JOB_RUNNING:
            st.info("집필하고 있어. 브라우저를 닫아도 작업은 계속돼.")
        elif status == JOB_FAILED_RECOVERABLE:
            st.warning(str(job.get("error_public") or "저장된 섹션부터 다시 이어 쓸 수 있어."))
        else:
            st.error(str(job.get("error_public") or "생성하지 못했어. 다시 요청해줘."))


@st.fragment(run_every=2.0)
def _story_live(config: AppConfig, store: ConsumerStore, user_id: str, story_id: str) -> None:
    story = store.get_owned_story(user_id, story_id)
    if story is None:
        st.error("작품을 찾을 수 없거나 보관 기간이 끝났어.")
        return
    workspace = StoryWorkspace.for_story(config, story_id, create=True)
    draft = read_draft(workspace.draft)
    sections = split_sections(draft)
    jobs = list(reversed(store.list_owned_jobs(user_id, story_id)))
    outstanding = store.owned_outstanding_job(user_id, story_id)
    queue = store.queue_stats()
    worker_label, worker_fresh = _worker_state(store)
    model = active_model_status(config, verify_files=False)

    progress = min(1.0, len(draft) / max(1, int(story["target_chars"])))
    completed = len(draft) >= int(story["target_chars"])
    outline = load_story_outline(workspace.outline)
    if outline and outline.beats:
        active_beat_index = min(
            len(outline.beats) - 1,
            int(progress * len(outline.beats)),
        )
        active_beat = outline.beats[active_beat_index]
        phase = html.escape(active_beat.phase)
        purpose = html.escape(active_beat.purpose)
        st.markdown(
            '<div class="outline-strip">'
            f'<div class="outline-label">현재 흐름 · {phase}</div>'
            f'<div class="outline-text">{purpose}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    metric_columns = st.columns(4)
    metric_columns[0].metric("진행률", f"{progress * 100:.1f}%")
    metric_columns[1].metric("분량", f"{len(draft):,} / {int(story['target_chars']):,}자")
    metric_columns[2].metric("섹션", f"{len(sections)}개")
    metric_columns[3].metric(
        "집필 상태",
        "완결" if completed else (worker_label if worker_fresh else "연결 확인 중"),
    )
    st.progress(progress)

    if jobs:
        for job in jobs:
            _render_job(job, sections, store, user_id)
    else:
        with st.chat_message("assistant"):
            st.markdown("첫 장면에서 일어날 사건이나 원하는 분위기를 말해줘.")

    st.divider()
    control_a, control_b = st.columns([1.3, 1])
    with control_a:
        creativity_label = st.segmented_control(
            "창의성",
            options=list(CREATIVITY_LABELS),
            default="균형",
            selection_mode="single",
            key="consumer_creativity",
        )
    with control_b:
        turn_chars = st.selectbox(
            "이번에 생성할 글자 수",
            options=config.consumer.allowed_turn_chars,
            index=config.consumer.allowed_turn_chars.index(config.consumer.default_turn_chars),
            format_func=lambda value: f"약 {value:,}자",
            key="consumer_turn_chars",
        )

    blocked_reason = ""
    if store.is_maintenance():
        blocked_reason = "서비스 점검 중이야. 진행 중인 원고는 안전하게 보관돼."
    elif not model["ready"]:
        blocked_reason = "집필 모델을 준비하고 있어. 관리자가 승격을 마치면 요청할 수 있어."
    elif outstanding:
        blocked_reason = f"현재 요청이 {STATUS_LABELS.get(str(outstanding['status']), '처리 중')}이야."
    elif not worker_fresh:
        blocked_reason = "집필 worker 연결을 확인하고 있어."
    elif completed:
        blocked_reason = "결말까지 완성했어. 전체 원고나 이어쓰기 번들을 내려받을 수 있어."
    if blocked_reason:
        st.caption(blocked_reason)

    prompt = st.chat_input(
        "다음 전개를 알려줘",
        disabled=bool(blocked_reason),
        key="consumer_chat_input",
    )
    if prompt:
        try:
            verified_model = active_model_status(config, verify_files=True)
            if not verified_model["ready"]:
                raise ConsumerStoreError(
                    "집필 모델 검증이 끝나지 않았어. 관리자가 모델 상태를 확인해야 해."
                )
            store.enqueue_job(
                user_id,
                story_id,
                instruction=prompt,
                creativity_profile=CREATIVITY_LABELS.get(str(creativity_label), "balanced"),
                requested_chars=int(turn_chars),
            )
            st.rerun(scope="fragment")
        except (ValueError, ConsumerStoreError) as exc:
            st.error(str(exc))

    if queue["queued"] or queue["running"]:
        st.caption(f"전체 대기 {queue['queued']}건 · 실행 {queue['running']}건")

    st.divider()
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "전체 원고",
        data=draft.encode("utf-8"),
        file_name=f"{story['title']}.md",
        mime="text/markdown",
        disabled=not bool(draft),
        width="stretch",
    )
    download_b.download_button(
        "이어쓰기 번들",
        data=build_continuation_bundle(workspace, story),
        file_name=f"{story['title']}_continue.zip",
        mime="application/zip",
        width="stretch",
    )


def _story_management(
    store: ConsumerStore,
    user: dict[str, Any],
    story: dict[str, Any],
) -> None:
    with st.sidebar:
        st.caption("로그인 계정")
        st.subheader(str(user["display_name"]))
        st.caption(f"@{user['username']}")
        if st.button("로그아웃", width="stretch", key="consumer_logout"):
            _logout(store)
        st.divider()
        st.subheader(str(story["title"]))
        st.caption(f"{story['genre']} · {int(story['target_chars']):,}자 목표")
        if st.button("내 작품으로", width="stretch"):
            _clear_story_session()
            st.rerun()
        with st.expander("작품 관리"):
            confirmation = st.checkbox("이 작품과 원고를 즉시 삭제할게", key="delete_story_confirmation")
            if st.button(
                "작품 삭제",
                disabled=not confirmation,
                type="secondary",
                width="stretch",
            ):
                store.delete_owned_story(str(user["id"]), str(story["id"]))
                _clear_story_session()
                st.rerun()
        st.caption(f"보관 만료 · {str(story['expires_at'])[:10]}")


def main() -> None:
    _styles()
    config, store = _load_runtime()
    user = _current_user(store)
    if user is None:
        _auth_entry(config, store)
        return

    story_id = str(st.session_state.get("consumer_story_id", ""))
    story = store.get_owned_story(str(user["id"]), story_id) if story_id else None
    if story is None:
        _clear_story_session()
        _story_entry(config, store, user)
        return

    _story_management(store, user, story)
    st.markdown('<div class="story-kicker">IN PROGRESS</div>', unsafe_allow_html=True)
    st.title(str(story["title"]))
    st.markdown(
        f'<div class="story-meta">{html.escape(str(story["genre"]))} · '
        '섹션을 이어 쓰며 완성하는 장편</div>',
        unsafe_allow_html=True,
    )
    _story_live(config, store, str(user["id"]), str(story["id"]))


if __name__ == "__main__":
    main()
