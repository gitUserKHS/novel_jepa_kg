from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cookie_context(cookies: dict[str, str]):
    """Patch the resolver behind `st.context.cookies`.

    AppTest replaces the runtime with a mock and offers no way to set request
    cookies, so the private resolver is the only seam. If a Streamlit upgrade
    moves it, this patch fails loudly rather than silently passing.
    """
    return patch(
        "streamlit.runtime.context._get_client_context",
        return_value=SimpleNamespace(cookies=cookies, headers={}),
    )


class SessionCookieContractTests(unittest.TestCase):
    """The refresh fix depends on a private Streamlit seam; pin its shape."""

    def test_the_patched_resolver_reaches_st_context_cookies(self) -> None:
        import streamlit as st

        with cookie_context({"novel_jepa_session": "token-123"}):
            self.assertEqual(st.context.cookies.get("novel_jepa_session"), "token-123")
            self.assertEqual(dict(st.context.cookies), {"novel_jepa_session": "token-123"})

    def test_a_missing_cookie_reads_as_the_default(self) -> None:
        import streamlit as st

        with cookie_context({}):
            self.assertEqual(st.context.cookies.get("novel_jepa_session", ""), "")


class CookieValueGuardTests(unittest.TestCase):
    """`_current_user` must only accept a real string as a session token."""

    def setUp(self) -> None:
        source = (PROJECT_ROOT / "consumer_app.py").read_text(encoding="utf-8")
        self.source = source

    def test_the_cookie_name_is_stable(self) -> None:
        self.assertIn('_SESSION_COOKIE = "novel_jepa_session"', self.source)

    def test_a_non_string_cookie_value_is_rejected(self) -> None:
        # Outside a browser st.context can return a stand-in object; str() on it
        # would produce a truthy pseudo-token.
        self.assertIn("isinstance(stored, str)", self.source)

    def test_the_token_never_travels_in_the_url(self) -> None:
        """A session token in a query string leaks through history and links."""
        for marker in (
            "query_params[_SESSION_COOKIE]",
            'query_params["novel_jepa_session"]',
            "query_params.novel_jepa_session",
        ):
            self.assertNotIn(marker, self.source)

    def test_the_cookie_is_scoped_and_expiring(self) -> None:
        self.assertIn("path=/", self.source)
        self.assertIn("max-age=", self.source)
        self.assertIn("SameSite=Lax", self.source)

    def test_logout_clears_the_cookie_on_the_following_run(self) -> None:
        self.assertIn('st.session_state["consumer_logout_pending"] = True', self.source)
        self.assertIn('st.session_state.pop("consumer_logout_pending", False)', self.source)
        self.assertIn("_forget_session_cookie()", self.source)


if __name__ == "__main__":
    unittest.main()
