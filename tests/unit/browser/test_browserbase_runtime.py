"""
Unit tests for BrowserBaseRuntime.

The runtime depends on both the ``browserbase`` SDK (sync) and Playwright
(async CDP). Both are mocked so no real credentials or browser are needed.

All patches target module-level references in ``theriad.browser.browserbase_runtime``
(_Browserbase, _async_playwright, _load_deps) so neither SDK needs to be installed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from theriad.browser.browserbase_runtime import BrowserBaseRuntime, BrowserBaseSession

# Patch target prefixes
_MOD = "theriad.browser.browserbase_runtime"


# ── Helpers ───────────────────────────────────────────────────────────────────

_DOM_RESPONSE = {
    "visible_text": "Hello BrowserBase",
    "clickable_elements": [{"id": "ae-1", "tag": "a", "text": "Click me", "href": "https://example.com/next"}],
    "input_fields": [],
    "accessibility_tree_summary": [{"role": "main", "name": "Main"}],
}


def _make_mock_page(url: str = "https://example.com/") -> AsyncMock:
    page = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value="Example")
    page.evaluate = AsyncMock(return_value=_DOM_RESPONSE)
    page.goto = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n")
    page.locator = MagicMock(return_value=AsyncMock())
    page.wait_for_timeout = AsyncMock()
    return page


def _make_playwright_stack(url: str = "https://example.com/") -> tuple:
    """Return (mock_pw, mock_browser, mock_context, mock_page)."""
    mock_page = _make_mock_page(url)
    mock_context = MagicMock()
    mock_context.pages = [mock_page]
    mock_browser = AsyncMock()
    mock_browser.contexts = [mock_context]
    mock_browser.disconnect = AsyncMock()
    mock_pw = AsyncMock()
    mock_pw.chromium = AsyncMock()
    mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()
    return mock_pw, mock_browser, mock_context, mock_page


def _make_bb_session(connect_url: str = "wss://fake.bb.com/session/abc123") -> MagicMock:
    bb_session = MagicMock()
    bb_session.id = "bb-session-xyz"
    bb_session.connect_url = connect_url
    return bb_session


def _make_bb_sdk(bb_session: MagicMock | None = None) -> MagicMock:
    bb_session = bb_session or _make_bb_session()
    bb = MagicMock()
    bb.sessions = MagicMock()
    bb.sessions.create = MagicMock(return_value=bb_session)
    return bb


def _rt(api_key: str = "test-key", project_id: str = "") -> BrowserBaseRuntime:
    return BrowserBaseRuntime(api_key=api_key, project_id=project_id)


# ── Availability ──────────────────────────────────────────────────────────────

class TestBrowserBaseRuntimeAvailability:

    def test_available_when_all_deps_present(self):
        rt = _rt(api_key="real-key")
        with patch(f"{_MOD}._load_deps", return_value=(True, None)):
            result = rt.availability()
        assert result["available"] is True
        assert result["error"] is None

    def test_unavailable_when_browserbase_missing(self):
        rt = _rt(api_key="real-key")
        with patch(f"{_MOD}._load_deps", return_value=(False, "browserbase SDK not installed")):
            result = rt.availability()
        assert result["available"] is False
        assert "browserbase" in result["error"]

    def test_unavailable_when_api_key_missing(self):
        rt = _rt(api_key="")
        with patch(f"{_MOD}._load_deps", return_value=(True, None)):
            result = rt.availability()
        assert result["available"] is False
        assert "BROWSERBASE_API_KEY" in result["error"]

    def test_unavailable_when_playwright_missing(self):
        rt = _rt(api_key="real-key")
        with patch(f"{_MOD}._load_deps", return_value=(False, "Playwright not installed")):
            result = rt.availability()
        assert result["available"] is False
        assert "Playwright" in result["error"]


# ── Session lifecycle ─────────────────────────────────────────────────────────

class TestBrowserBaseRuntimeSession:

    async def test_start_session_success(self):
        rt = _rt()
        mock_pw, mock_browser, _, mock_page = _make_playwright_stack()
        bb = _make_bb_sdk()

        with (
            patch(f"{_MOD}.BrowserBaseRuntime.availability",
                  return_value={"available": True, "error": None}),
            patch(f"{_MOD}._Browserbase", return_value=bb),
            patch(f"{_MOD}._async_playwright", _async_playwright_cm(mock_pw)),
        ):
            result = await rt.start_session(session_id="s1")

        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["backend"] == "browserbase"
        assert result["bb_session_id"] == "bb-session-xyz"
        assert "s1" in rt.sessions

    async def test_start_session_unavailable(self):
        rt = _rt(api_key="")
        with patch(f"{_MOD}._load_deps", return_value=(True, None)):
            result = await rt.start_session(session_id="s1")
        assert result["ok"] is False
        assert "s1" not in rt.sessions

    async def test_start_session_reuses_existing(self):
        rt = _rt()
        mock_pw, _, _, mock_page = _make_playwright_stack()
        rt.sessions["s1"] = BrowserBaseSession(
            session_id="s1",
            bb_session_id="existing-bb",
            playwright=mock_pw,
            browser=AsyncMock(),
            context=MagicMock(),
            page=mock_page,
        )
        result = await rt.status("s1")
        assert result["ok"] is True
        assert result["bb_session_id"] == "existing-bb"

    async def test_stop_session_removes_and_disconnects(self):
        rt = _rt()
        mock_pw, mock_browser, _, mock_page = _make_playwright_stack()
        rt.sessions["s1"] = BrowserBaseSession(
            session_id="s1",
            bb_session_id="bb-abc",
            playwright=mock_pw,
            browser=mock_browser,
            context=MagicMock(),
            page=mock_page,
        )
        result = await rt.stop_session("s1")
        assert result["ok"] is True
        assert "s1" not in rt.sessions
        mock_browser.disconnect.assert_called_once()
        mock_pw.stop.assert_called_once()

    async def test_stop_unknown_session(self):
        rt = _rt()
        result = await rt.stop_session("ghost")
        assert result["ok"] is False
        assert result["status"] == "not_found"

    async def test_stop_all(self):
        rt = _rt()
        mock_pw, mock_browser, _, page = _make_playwright_stack()
        for sid in ("a", "b"):
            rt.sessions[sid] = BrowserBaseSession(
                session_id=sid,
                bb_session_id=f"bb-{sid}",
                playwright=mock_pw,
                browser=mock_browser,
                context=MagicMock(),
                page=page,
            )
        await rt.stop_all()
        assert len(rt.sessions) == 0


# ── Observation ───────────────────────────────────────────────────────────────

class TestBrowserBaseRuntimeObserve:

    @pytest.fixture()
    def rt_with_session(self):
        rt = _rt()
        _, _, _, mock_page = _make_playwright_stack()
        rt.sessions["s1"] = BrowserBaseSession(
            session_id="s1",
            bb_session_id="bb-obs",
            playwright=AsyncMock(),
            browser=AsyncMock(),
            context=MagicMock(),
            page=mock_page,
        )
        return rt

    async def test_observe_returns_expected_keys(self, rt_with_session):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt_with_session.observe("s1")
        assert result["ok"] is True
        assert result["backend"] == "browserbase"
        assert result["visible_text_summary"] == "Hello BrowserBase"
        assert result["current_url"] == "https://example.com/"
        assert result["page_title"] == "Example"
        assert result["clickable_elements"][0]["id"] == "ae-1"
        assert result["screenshot_base64"] is None  # screenshot=False by default

    async def test_observe_with_screenshot(self, rt_with_session):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt_with_session.observe("s1", screenshot=True)
        assert result["screenshot_base64"] is not None

    async def test_observe_unknown_session_raises(self, rt_with_session):
        with pytest.raises(KeyError):
            await rt_with_session.observe("ghost")

    async def test_observe_captcha_detection(self, rt_with_session):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(True, "Cloudflare CAPTCHA detected"),
        ):
            result = await rt_with_session.observe("s1")
        assert result["captcha_detected"] is True
        assert result["captcha_reason"] == "Cloudflare CAPTCHA detected"


# ── Actions ───────────────────────────────────────────────────────────────────

class TestBrowserBaseRuntimeActions:

    @pytest.fixture()
    def rt(self):
        rt = _rt()
        _, _, _, mock_page = _make_playwright_stack()
        rt.sessions["s1"] = BrowserBaseSession(
            session_id="s1",
            bb_session_id="bb-act",
            playwright=AsyncMock(),
            browser=AsyncMock(),
            context=MagicMock(),
            page=mock_page,
        )
        return rt

    async def test_navigate_action(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt.action("s1", {"action": "navigate", "url": "https://example.com/"})
        assert result["ok"] is True
        assert result["backend"] == "browserbase"

    async def test_click_action(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt.action("s1", {"action": "click", "element_id": "ae-1"})
        assert result["ok"] is True

    async def test_fill_action(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt.action("s1", {"action": "fill", "element_id": "ae-2", "text": "hello"})
        assert result["ok"] is True

    async def test_scroll_action(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt.action("s1", {"action": "scroll", "amount": 500})
        assert result["ok"] is True

    async def test_wait_action(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            result = await rt.action("s1", {"action": "wait", "milliseconds": 10})
        assert result["ok"] is True

    async def test_unsupported_action(self, rt):
        result = await rt.action("s1", {"action": "hover"})
        assert result["ok"] is False
        assert "Unsupported" in result["error"]

    async def test_missing_action_key(self, rt):
        result = await rt.action("s1", {})
        assert result["ok"] is False

    async def test_action_logs_recorded(self, rt):
        with patch(
            "theriad.browser.browserbase_runtime.detect_captcha",
            return_value=(False, None),
        ):
            await rt.action("s1", {"action": "scroll", "amount": 200})
        log = rt.sessions["s1"].action_log
        assert any(e["action"] == "scroll" and e["status"] == "executed" for e in log)


# ── Status ────────────────────────────────────────────────────────────────────

class TestBrowserBaseRuntimeStatus:

    @pytest.fixture()
    def rt(self):
        rt = _rt()
        _, _, _, mock_page = _make_playwright_stack()
        rt.sessions["s1"] = BrowserBaseSession(
            session_id="s1",
            bb_session_id="bb-st",
            playwright=AsyncMock(),
            browser=AsyncMock(),
            context=MagicMock(),
            page=mock_page,
        )
        return rt

    async def test_status_known_session(self, rt):
        result = await rt.status("s1")
        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["backend"] == "browserbase"
        assert result["bb_session_id"] == "bb-st"

    async def test_status_unknown_session(self, rt):
        result = await rt.status("ghost")
        assert result["ok"] is False
        assert result["status"] == "not_found"

    async def test_global_status_with_active_sessions(self, rt):
        result = await rt.status()
        assert result["active_sessions"] == 1
        assert result["backend"] == "browserbase"


# ── Helpers (test-local) ──────────────────────────────────────────────────────

def _async_playwright_cm(mock_pw: AsyncMock):
    """
    Return a callable that mimics ``async_playwright()`` used as:
        pw = await async_playwright().start()
    """
    inner = MagicMock()
    inner.start = AsyncMock(return_value=mock_pw)
    factory = MagicMock(return_value=inner)
    return factory
