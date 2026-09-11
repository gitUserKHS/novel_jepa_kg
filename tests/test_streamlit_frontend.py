from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_isolated_scenario(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--scenario", name],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Isolated Streamlit scenario {name!r} failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


class StreamlitFrontendTests(unittest.TestCase):
    def test_dashboard_renders_with_service_status(self) -> None:
        _run_isolated_scenario("dashboard")

    def test_token_login_unlocks_dashboard(self) -> None:
        _run_isolated_scenario("admin_token")

    def test_admin_keeps_research_controls_and_service_management(self) -> None:
        _run_isolated_scenario("admin_controls")

    def test_consumer_plans_and_starts_a_story(self) -> None:
        _run_isolated_scenario("consumer")

    def test_consumer_manual_template_planning(self) -> None:
        _run_isolated_scenario("consumer_manual_template")

    def test_consumer_section_editing_refreshes_memory(self) -> None:
        _run_isolated_scenario("consumer_edit_section")


def _scenario_dashboard() -> None:
    from streamlit.testing.v1 import AppTest

    environment = {
        "NOVEL_JEPA_CHAT_MODEL": "gemma4:e4b",
        "NOVEL_JEPA_OLLAMA_BASE_URL": "http://127.0.0.1:9",
        "NOVEL_JEPA_REQUIRE_AUTH": "false",
    }
    with patch.dict(os.environ, environment, clear=False):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert len(app.exception) == 0, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Service"] == "Local host"
    assert metrics["Base model"] == "gemma4:e4b"
    assert metrics["Access"] == "Local only"


def _scenario_admin_token() -> None:
    from streamlit.testing.v1 import AppTest

    environment = {
        "NOVEL_JEPA_CHAT_MODEL": "gemma4:e4b",
        "NOVEL_JEPA_OLLAMA_BASE_URL": "http://127.0.0.1:9",
        "NOVEL_JEPA_REQUIRE_AUTH": "true",
        "NOVEL_JEPA_ACCESS_TOKEN": "test-only-secret",
    }
    with patch.dict(os.environ, environment, clear=False):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        app.text_input[0].set_value("test-only-secret")
        next(button for button in app.button if button.label == "로그인").click()
        app.run()

    assert len(app.exception) == 0, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Access"] == "Token"


def _scenario_admin_controls() -> None:
    from streamlit.testing.v1 import AppTest

    environment = {
        "NOVEL_JEPA_OLLAMA_BASE_URL": "http://127.0.0.1:9",
        "NOVEL_JEPA_REQUIRE_AUTH": "false",
    }
    with tempfile.TemporaryDirectory() as temporary, patch.dict(
        os.environ,
        {**environment, "NOVEL_JEPA_OUTPUT_ROOT": temporary},
        clear=False,
    ):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    labels = [tab.label for tab in app.tabs]
    assert len(app.exception) == 0, app.exception
    for required in ["Dataset", "Embedding", "Train", "Evaluate", "Service"]:
        assert required in labels


def _scenario_consumer() -> None:
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory() as temporary, patch.dict(
        os.environ,
        {"NOVEL_JEPA_OUTPUT_ROOT": temporary, "NOVEL_LLM_DRY_RUN": "1"},
        clear=False,
    ):
        app = AppTest.from_file(str(PROJECT_ROOT / "consumer_app.py"), default_timeout=30).run()
        assert [tab.label for tab in app.tabs] == ["로그인", "회원가입"]

        next(item for item in app.text_input if item.label == "표시 이름").set_value("소비자")
        next(item for item in app.text_input if item.label == "새 아이디").set_value("reader01")
        next(item for item in app.text_input if item.label == "새 비밀번호").set_value("reader-password")
        next(item for item in app.text_input if item.label == "비밀번호 확인").set_value("reader-password")
        next(button for button in app.button if button.label == "회원가입").click()
        app.run()
        assert len(app.exception) == 0, app.exception

        # 로그인 직후 바로 장편 소설 기획 화면이다: 모드 전환·말투·일반 채팅은 없다.
        assert len(app.radio) == 0, "no mode switch any more"
        assert not any(item.label == "말투" for item in app.selectbox)
        assert any(title.value == "어떤 이야기를 쓰고 싶어?" for title in app.title)
        assert [tab.label for tab in app.tabs] == ["💬 대화로 기획", "📝 직접 작성"]
        assert len(app.chat_input) == 1
        visible = [button.label for button in app.button] + [tab.label for tab in app.tabs]
        for forbidden in ["Dataset", "Embedding", "Train", "Evaluate", "Service", "➕ 새 대화"]:
            assert forbidden not in visible
        assert any("모델 준비됨" in item.value for item in app.caption), [c.value for c in app.caption]

        # 기획 대화 한 턴 (dry-run 모델): 작품 카드가 채워진다.
        app.chat_input[0].set_value("비 오는 도시의 미스터리, 주인공은 기록 복원가 서윤. 3만 자쯤.").run()
        assert len(app.exception) == 0, app.exception
        assert any("작품 카드" in item.value for item in app.markdown)
        start_button = next(button for button in app.button if button.label == "✅ 이 설정으로 집필 시작")
        assert not start_button.disabled
        target_selector = next(item for item in app.selectbox if item.label == "목표 분량")
        assert "42,000자" in target_selector.options
        start_button.click()
        app.run()
        assert len(app.exception) == 0, app.exception

        # 작품 화면: 제목·진행 지표·턴 분량 선택·첫 턴이 큐에 들어가 있다.
        assert any(title.value == "유리등의 속삭임" for title in app.title)
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["분량"] == "0 / 30,000자"
        turn_selector = next(item for item in app.selectbox if item.label == "이번에 생성할 글자 수")
        assert turn_selector.options == ["약 2,000자", "약 3,000자", "약 5,000자"]
        assert any("대기 순번" in item.value for item in app.markdown)


def _register(app, username: str):  # noqa: ANN001, ANN202
    next(item for item in app.text_input if item.label == "표시 이름").set_value("작가")
    next(item for item in app.text_input if item.label == "새 아이디").set_value(username)
    next(item for item in app.text_input if item.label == "새 비밀번호").set_value("reader-password")
    next(item for item in app.text_input if item.label == "비밀번호 확인").set_value("reader-password")
    next(button for button in app.button if button.label == "회원가입").click()
    app.run()
    assert len(app.exception) == 0, app.exception


def _button(app, label: str):  # noqa: ANN001, ANN202
    return next(b for b in app.button if b.label == label)


def _scenario_consumer_manual_template() -> None:
    """직접 작성 양식: 템플릿 저장 → 불러오기 → 집필 시작 → 작품 화면에 집필 지침 → 작품에서 템플릿 저장."""
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory() as temporary, patch.dict(
        os.environ,
        {"NOVEL_JEPA_OUTPUT_ROOT": temporary, "NOVEL_LLM_DRY_RUN": "1"},
        clear=False,
    ):
        app = AppTest.from_file(str(PROJECT_ROOT / "consumer_app.py"), default_timeout=30).run()
        _register(app, "writer03")
        assert [tab.label for tab in app.tabs] == ["💬 대화로 기획", "📝 직접 작성"]
        assert not any(item.label == "내 템플릿" for item in app.selectbox), "no templates yet"

        # 양식을 채우고 템플릿으로 저장
        values = {
            "tpl_title": "등대의 딸", "tpl_genre": "잔잔한 미스터리",
            "tpl_premise": "비 오는 항구 도시에서 아버지의 실종을 쫓는다.",
            "tpl_world": "안개가 걷히지 않는 남해안 항구 도시", "tpl_protagonist": "해원: 등대지기의 딸",
            "tpl_characters": "문 형사: 과거를 숨긴 형사", "tpl_style_guide": "1인칭, 짧은 문장, 매 장 끝에 여운",
        }
        for key, value in values.items():
            widget = app.text_input(key=key) if key in ("tpl_title", "tpl_genre") else app.text_area(key=key)
            widget.set_value(value)
        app.text_input(key="tpl_name").set_value("항구 세계관")
        _button(app, "💾 템플릿으로 저장").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert any("템플릿 '항구 세계관' 을 저장했어" in toast.value for toast in app.toast), [t.value for t in app.toast]
        picker = next(item for item in app.selectbox if item.label == "내 템플릿")
        assert picker.options == ["항구 세계관"]

        # 불러오면 양식이 채워진다 (저장 뒤 양식은 그대로이므로 먼저 비워서 확인)
        app.text_area(key="tpl_world").set_value("")
        _button(app, "불러오기").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert app.text_area(key="tpl_world").value == "안개가 걷히지 않는 남해안 항구 도시"
        assert app.text_area(key="tpl_style_guide").value == "1인칭, 짧은 문장, 매 장 끝에 여운"

        # 필수 항목이 비면 시작하지 않는다
        app.text_input(key="tpl_genre").set_value("")
        _button(app, "✅ 이 설정으로 집필 시작").click()
        app.run()
        assert any("비어 있는 항목" in item.value for item in app.error), [e.value for e in app.error]
        app.text_input(key="tpl_genre").set_value("잔잔한 미스터리")
        _button(app, "✅ 이 설정으로 집필 시작").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert any(title.value == "등대의 딸" for title in app.title)
        assert any("1인칭, 짧은 문장" in item.value for item in app.markdown), "settings view shows the style guide"
        assert any("대기 순번" in item.value for item in app.markdown), "the first turn is queued"
        # 집필 지침은 워커가 쓰는 설정 시트에 들어간다
        from src.service.consumer_store import ConsumerStore
        from src.service.story_sheets import style_guide, world_sheet
        from src.utils.config import load_config

        config = load_config("configs/default.yaml")
        store = ConsumerStore(config)
        owner = str(store.authenticate_user("writer03", "reader-password")["id"])
        story = store.list_owned_stories(owner)[0]
        assert story["style_guide"] == "1인칭, 짧은 문장, 매 장 끝에 여운"
        assert style_guide(story) == "1인칭, 짧은 문장, 매 장 끝에 여운"
        assert "집필 지침" not in world_sheet(story)

        # 작품 화면에서도 템플릿으로 저장할 수 있다 (같은 이름이면 덮어쓴다)
        next(item for item in app.text_input if item.label == "템플릿 이름").set_value("항구 세계관")
        _button(app, "💾 템플릿으로 저장").click()
        app.run()
        assert any("템플릿 '항구 세계관' 으로 저장했어" in item.value for item in app.success), [x.value for x in app.success]
        from src.service.template_store import StoryTemplateStore

        templates = StoryTemplateStore(config).list_templates(owner)
        assert [t["name"] for t in templates] == ["항구 세계관"]
        assert templates[0]["title"] == "등대의 딸"


def _scenario_consumer_edit_section() -> None:
    """원고 수정 패널: ✏️ 로 장 선택 → 직접 수정 저장(메모리 갱신) → AI 보조 제안 저장 → 메모리 직접 저장."""
    from streamlit.testing.v1 import AppTest

    from src.memory.story_rag import StoryMemory, load_story_memories, write_story_memories
    from src.service.consumer_store import ConsumerStore
    from src.service.story_workspace import StoryWorkspace
    from src.utils.config import load_config

    with tempfile.TemporaryDirectory() as temporary, patch.dict(
        os.environ,
        {"NOVEL_JEPA_OUTPUT_ROOT": temporary, "NOVEL_LLM_DRY_RUN": "1"},
        clear=False,
    ):
        app = AppTest.from_file(str(PROJECT_ROOT / "consumer_app.py"), default_timeout=30).run()
        _register(app, "writer04")

        # 끝난 턴이 두 장을 쓴 작품을 저장소에서 직접 만든다 (워커 없이).
        config = load_config("configs/default.yaml")
        store = ConsumerStore(config)
        owner = str(store.authenticate_user("writer04", "reader-password")["id"])
        story = store.create_story(
            owner, title="유리등의 속삭임", genre="SF 미스터리", premise="사라진 기억을 추적한다",
            world="기억이 거래되는 근미래 서울", protagonist="서윤: 기록 복원가", characters="민재: 연구원",
            target_chars=10000,
        )
        sid = str(story["id"])
        workspace = StoryWorkspace.for_story(config, sid, create=True)
        workspace.draft.write_text(
            "### 1장 첫 신호\n\n서윤은 장치를 켰다. 민재는 말이 없었다.\n\n"
            "### 2장 심층 구역\n\n서윤은 좌표를 따라 내려갔다. 문이 열려 있었다.",
            encoding="utf-8",
        )
        write_story_memories(workspace.memory, [
            StoryMemory(section_index=1, title="1장 첫 신호", summary="옛 요약 하나"),
            StoryMemory(section_index=2, title="2장 심층 구역", summary="옛 요약 둘"),
        ])
        job = store.enqueue_job(owner, sid, instruction="첫 장면을 써줘.", requested_chars=2000, creativity_profile="balanced")
        store.claim_next_job("worker-1", "v1")
        store.set_job_start_section_count(int(job["id"]), 0)
        store.complete_job(int(job["id"]), result_chars=60, result_section_count=2, total_chars=60,
                           total_section_count=2, metrics={}, novel_completed=False)

        app.run()
        next(b for b in app.sidebar.button if b.label == "유리등의 속삭임").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert any(title.value == "유리등의 속삭임" for title in app.title)

        # 본문 아래의 ✏️ 로 2장을 고른다.
        edit_buttons = [b for b in app.button if b.label == "✏️ 이 장 수정"]
        assert len(edit_buttons) == 2, [b.key for b in app.button]
        edit_buttons[1].click()
        app.run()
        assert len(app.exception) == 0, app.exception
        picker = app.selectbox(key="consumer_edit_section")
        assert picker.value == 2, picker.value

        def area(label: str):  # noqa: ANN202
            return next(item for item in app.text_area if item.label == label)

        body_label = "본문 (첫 줄은 '### 소제목')"
        assert area(body_label).value.startswith("### 2장 심층 구역")

        # 직접 수정 + 메모리 갱신 (dry-run 모델은 고정 JSON 을 돌려준다)
        area(body_label).set_value("### 2장 심층 구역\n\n서윤은 좌표를 따라 내려갔다. 문은 잠겨 있었고, 안에서 목소리가 들렸다.")
        _button(app, "💾 저장하고 메모리 갱신").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        draft = workspace.draft.read_text(encoding="utf-8")
        assert "문은 잠겨 있었고" in draft and "문이 열려 있었다" not in draft
        assert "서윤은 장치를 켰다" in draft, "section 1 untouched"
        memories = load_story_memories(workspace.memory)
        assert memories[0].summary == "옛 요약 하나"
        assert memories[1].summary == "서윤이 첫 단서를 얻는다.", memories[1].summary
        assert any("2장을 저장했어" in toast.value for toast in app.toast), [t.value for t in app.toast]
        assert int(store.get_owned_story(owner, sid)["current_chars"]) == len(draft.strip())
        # 저장 뒤 요약 메모리 탭의 입력칸도 새 메모리를 보여야 한다 (옛 값이 남아 덮어쓰면 안 됨)
        assert area("한 줄 요약").value == "서윤이 첫 단서를 얻는다.", area("한 줄 요약").value

        # AI 보조: 빠른 요청을 고르면 버튼이 열리고, 제안이 따로 나온 뒤 저장할 수 있다.
        assert _button(app, "🤖 AI 에게 맡기기").disabled
        app.pills(key=f"edit_quick_{sid}_2").set_value("더 긴장감 있게")
        app.run()
        assert not _button(app, "🤖 AI 에게 맡기기").disabled
        _button(app, "🤖 AI 에게 맡기기").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        proposal = area("AI 제안 — 더 고쳐서 저장할 수 있어")
        assert proposal.value.startswith("### 젖은 골목의 신호"), proposal.value[:60]
        assert "문은 잠겨 있었고" in workspace.draft.read_text(encoding="utf-8"), "proposal alone changes nothing"
        _button(app, "💾 이 내용으로 저장").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        draft = workspace.draft.read_text(encoding="utf-8")
        assert "### 젖은 골목의 신호" in draft and "문은 잠겨 있었고" not in draft
        assert "서윤은 장치를 켰다" in draft

        # 요약 메모리 직접 저장
        area("한 줄 요약").set_value("서윤이 심층 구역에서 동생의 목소리를 들었다.")
        area("확정된 사실 (줄마다 하나)").set_value("좌표는 심층 구역을 가리켰다\n문은 잠겨 있었다")
        _button(app, "💾 메모리 저장").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        memories = load_story_memories(workspace.memory)
        assert memories[1].summary == "서윤이 심층 구역에서 동생의 목소리를 들었다."
        assert memories[1].facts == ["좌표는 심층 구역을 가리켰다", "문은 잠겨 있었다"]
        assert any("메모리를 저장했어" in toast.value for toast in app.toast)

        # 자동 이어쓰기: 켜면 기다리는 턴이 없을 때 첫 자동 턴이 바로 들어가고, 끄면 그 턴은 남는다.
        from src.service.consumer_store import JOB_ORIGIN_AUTO

        _button(app, "🤖 자동 이어쓰기 켜기").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert store.get_owned_story(owner, sid)["auto_continue"] is True
        running = store.owned_outstanding_job(owner, sid)
        assert running is not None and running["origin"] == JOB_ORIGIN_AUTO, running
        assert any("자동 이어쓰기 켜짐" in item.value for item in app.markdown)
        assert any("🤖 AI 가 다음 전개를 정해 이어 써." in item.value for item in app.markdown), "the auto turn shows as a bubble"
        _button(app, "자동 이어쓰기 끄기").click()
        app.run()
        assert store.get_owned_story(owner, sid)["auto_continue"] is False
        assert store.owned_outstanding_job(owner, sid)["id"] == running["id"], "switching off keeps the queued turn"
        assert _button(app, "🤖 다음 한 턴만 AI 에게").disabled, "one turn is already outstanding"

        # 집필 요청이 들어가 있으면 패널은 잠긴다.
        assert any("끝난 뒤에 고칠 수 있어" in item.value for item in app.warning), [w.value for w in app.warning]
        assert _button(app, "💾 저장하고 메모리 갱신").disabled

        # 게이트가 장을 다시 쓰는 동안: 사유와 버려진 초안이 화면에 남는다 (사라진 것처럼 보이지 않게).
        from src.service.story_workspace import LiveProseWriter

        store.claim_next_job("worker-1", "v1")
        store.set_job_start_section_count(int(running["id"]), 2)
        writer = LiveProseWriter(workspace, flush_chars=1)
        writer.begin_section()
        writer.feed("### 3장 첫 초안" + chr(10) * 2 + "같은 문장이 계속 돌았다. 같은 문장이 계속 돌았다.")
        writer.restart_section("같은 구절이 6회 반복됨")
        writer.feed("### 3장 다시 쓴 판" + chr(10) * 2 + "서윤은 문을 열었다.")
        writer.flush()
        app.run()
        assert len(app.exception) == 0, app.exception
        assert any("같은 구절이 6회 반복됨 — 이 장을 다시 쓰는 중이야" in item.value for item in app.warning), [w.value for w in app.warning]
        assert any(item.label == "버려진 초안 보기 (저장되지 않음)" for item in app.expander)
        assert any("같은 문장이 계속 돌았다" in item.value for item in app.markdown)
        assert any("서윤은 문을 열었다" in item.value for item in app.markdown), "the rewrite streams beneath the note"

        # 끝난 턴에는 기록된 사유가 남고, 자동 턴은 🤖 로 표시된다 (워커가 정한 지시로 바뀐 뒤에도).
        writer.commit_section()
        store.set_job_instruction(int(running["id"]), "민재가 서윤을 막아서게 해줘.")
        store.complete_job(int(running["id"]), result_chars=20, result_section_count=1, total_chars=80,
                           total_section_count=3, metrics={"retry_notes": ["3장: 같은 구절이 6회 반복됨", "3장: 반복 문장 4개를 걷어냄 (다시 쓰지 않음)"]},
                           novel_completed=False)
        store.heartbeat_worker("worker-1", "idle")  # 워커가 살아 있어야 새 턴 버튼이 열린다
        app.run()
        captions = [item.value for item in app.caption]
        assert any("🔁 3장: 같은 구절이 6회 반복됨" in value for value in captions), captions
        assert any("✂️ 3장: 반복 문장 4개를 걷어냄" in value for value in captions), captions
        assert any(item.value == "🤖 민재가 서윤을 막아서게 해줘." for item in app.markdown), [m.value for m in app.markdown][-12:]
        # 기다리는 턴이 없으니 '한 턴만' 이 열리고, 누르면 자동 턴이 들어간다.
        assert not _button(app, "🤖 다음 한 턴만 AI 에게").disabled
        _button(app, "🤖 다음 한 턴만 AI 에게").click()
        app.run()
        assert len(app.exception) == 0, app.exception
        queued = store.owned_outstanding_job(owner, sid)
        assert queued is not None and queued["origin"] == JOB_ORIGIN_AUTO and queued["instruction"] == "🤖 AI 가 다음 전개를 정해 이어 써."
        assert store.get_owned_story(owner, sid)["auto_continue"] is False, "one-off turn does not switch the chain on"


SCENARIOS = {
    "dashboard": _scenario_dashboard,
    "admin_token": _scenario_admin_token,
    "admin_controls": _scenario_admin_controls,
    "consumer": _scenario_consumer,
    "consumer_manual_template": _scenario_consumer_manual_template,
    "consumer_edit_section": _scenario_consumer_edit_section,
}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        scenario = SCENARIOS.get(sys.argv[2])
        if scenario is None:
            raise SystemExit(f"Unknown scenario: {sys.argv[2]}")
        scenario()
    else:
        unittest.main()
