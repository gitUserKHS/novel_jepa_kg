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

    def test_consumer_registers_and_has_only_story_and_chat_surfaces(self) -> None:
        _run_isolated_scenario("consumer")


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
        {"NOVEL_JEPA_OUTPUT_ROOT": temporary},
        clear=False,
    ):
        app = AppTest.from_file(str(PROJECT_ROOT / "consumer_app.py"), default_timeout=30).run()
        assert [tab.label for tab in app.tabs] == ["로그인", "회원가입"]
        assert not any(item.label == "초대코드" for item in app.text_input)
        assert not any(item.label == "작품키" for item in app.text_input)

        next(item for item in app.text_input if item.label == "표시 이름").set_value("소비자")
        next(item for item in app.text_input if item.label == "새 아이디").set_value("reader01")
        next(item for item in app.text_input if item.label == "새 비밀번호").set_value(
            "reader-password"
        )
        next(item for item in app.text_input if item.label == "비밀번호 확인").set_value(
            "reader-password"
        )
        next(button for button in app.button if button.label == "회원가입").click()
        app.run()

        assert len(app.exception) == 0, app.exception
        assert [tab.label for tab in app.tabs] == ["내 작품", "새 작품"]
        visible_labels = [tab.label for tab in app.tabs] + [button.label for button in app.button]
        for forbidden in ["Dataset", "Embedding", "Train", "Evaluate"]:
            assert forbidden not in visible_labels

        next(item for item in app.text_input if item.label == "제목").set_value("소비자 테스트")
        next(item for item in app.text_input if item.label == "장르").set_value("SF 미스터리")
        next(item for item in app.text_area if item.label == "핵심 소재").set_value("사라진 기억")
        next(item for item in app.text_area if item.label == "세계관").set_value(
            "기억이 거래되는 서울"
        )
        next(item for item in app.text_area if item.label == "주인공").set_value(
            "서윤: 기록 복원가"
        )
        target_selector = next(
            item for item in app.selectbox if item.label == "전체 목표 글자 수"
        )
        assert "42,000자" in target_selector.options
        target_selector.set_value(42000)
        next(button for button in app.button if button.label == "작품 만들기").click()
        app.run()

        assert len(app.exception) == 0, app.exception
        assert any(title.value == "소비자 테스트" for title in app.title)
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["분량"] == "0 / 42,000자"
        turn_selector = next(
            item for item in app.selectbox if item.label == "이번에 생성할 글자 수"
        )
        assert turn_selector.options == ["약 2,000자", "약 3,000자", "약 5,000자"]
        labels_after_create = [button.label for button in app.button]
        for forbidden in ["Dataset", "Embedding", "Train", "Evaluate"]:
            assert forbidden not in labels_after_create


SCENARIOS = {
    "dashboard": _scenario_dashboard,
    "admin_token": _scenario_admin_token,
    "admin_controls": _scenario_admin_controls,
    "consumer": _scenario_consumer,
}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        scenario = SCENARIOS.get(sys.argv[2])
        if scenario is None:
            raise SystemExit(f"Unknown scenario: {sys.argv[2]}")
        scenario()
    else:
        unittest.main()
