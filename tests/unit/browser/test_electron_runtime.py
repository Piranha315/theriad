"""
Unit tests for ElectronBrowserRuntime.

The runtime speaks to the Electron RPC server over HTTP. These tests mock the
HTTP layer so no real Electron process is needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from theriad.browser.electron_runtime import ElectronBrowserRuntime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_urlopen(response_body: dict):
    """Return a context-manager mock that yields JSON bytes."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=json.dumps(response_body).encode())
    return cm


_OBSERVE_RESPONSE = {
    "ok": True,
    "backend": "electron",
    "url": "https://example.com/",
    "title": "Example Domain",
    "visible_text": "Example Domain\nThis domain is for use in illustrative examples.",
    "clickable_elements": [{"id": "ae-1", "tag": "a", "text": "More information...", "href": "https://www.iana.org/domains/reserved"}],
    "input_fields": [],
    "accessibility_tree_summary": [{"role": "main", "name": ""}],
    "screenshot_base64": None,
}

_HEALTH_RESPONSE = {"ok": True, "version": "1.0", "backend": "electron"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestElectronBrowserRuntimeAvailability:

    def test_available_when_health_returns_ok(self):
        rt = ElectronBrowserRuntime(port=47200)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_HEALTH_RESPONSE)):
            result = rt.availability()
        assert result["available"] is True
        assert result["error"] is None

    def test_unavailable_when_connection_refused(self):
        import urllib.error
        rt = ElectronBrowserRuntime(port=47200)
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = rt.availability()
        assert result["available"] is False
        assert "not running" in result["error"]

    def test_unavailable_when_health_returns_not_ok(self):
        rt = ElectronBrowserRuntime(port=47200)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({"ok": False})):
            result = rt.availability()
        assert result["available"] is False


class TestElectronBrowserRuntimeSession:

    @pytest.fixture()
    def rt(self):
        return ElectronBrowserRuntime(port=47200)

    @pytest.fixture()
    def available_rt(self, rt):
        """Runtime with health check patched to succeed."""
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_HEALTH_RESPONSE)):
            yield rt

    async def test_start_session_returns_ok(self, available_rt):
        rt = available_rt
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_HEALTH_RESPONSE)):
            result = await rt.start_session(session_id="s1", user_prompt="test")
        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["backend"] == "electron"
        assert "s1" in rt.sessions

    async def test_start_session_unavailable_returns_error(self, rt):
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            result = await rt.start_session(session_id="s1")
        assert result["ok"] is False
        assert "s1" not in rt.sessions

    async def test_start_session_navigates_to_start_url(self, rt):
        nav_response = {"ok": True, "url": "https://example.com/", "title": "Example"}
        responses = iter([_HEALTH_RESPONSE, nav_response, _OBSERVE_RESPONSE])

        def mock_urlopen(req, timeout=None):
            return _mock_urlopen(next(responses))

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = await rt.start_session(session_id="s2", start_url="https://example.com/")
        assert result["ok"] is True
        assert rt.sessions["s2"].current_url == "https://example.com/"

    async def test_stop_session_removes_session(self, rt):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_HEALTH_RESPONSE)):
            await rt.start_session(session_id="s3")
        result = await rt.stop_session("s3")
        assert result["ok"] is True
        assert "s3" not in rt.sessions

    async def test_stop_nonexistent_session_returns_not_found(self, rt):
        result = await rt.stop_session("ghost")
        assert result["ok"] is False
        assert result["status"] == "not_found"


class TestElectronBrowserRuntimeObserve:

    @pytest.fixture()
    def rt_with_session(self):
        rt = ElectronBrowserRuntime(port=47200)
        from theriad.browser.electron_runtime import ElectronSession
        rt.sessions["s1"] = ElectronSession(
            session_id="s1",
            current_url="https://example.com/",
        )
        return rt

    async def test_observe_returns_normalised_keys(self, rt_with_session):
        rt = rt_with_session
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_OBSERVE_RESPONSE)):
            result = await rt.observe("s1")
        assert result["ok"] is True
        assert result["page_title"] == "Example Domain"
        assert result["current_url"] == "https://example.com/"
        assert result["visible_text_summary"] == _OBSERVE_RESPONSE["visible_text"]
        assert result["backend"] == "electron"
        assert result["session_id"] == "s1"

    async def test_observe_updates_session_url(self, rt_with_session):
        rt = rt_with_session
        resp = {**_OBSERVE_RESPONSE, "url": "https://example.com/page2"}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp)):
            await rt.observe("s1")
        assert rt.sessions["s1"].current_url == "https://example.com/page2"

    async def test_observe_unknown_session_raises(self, rt_with_session):
        with pytest.raises(KeyError):
            await rt_with_session.observe("no-such-session")


class TestElectronBrowserRuntimeActions:

    @pytest.fixture()
    def rt(self):
        rt = ElectronBrowserRuntime(port=47200)
        from theriad.browser.electron_runtime import ElectronSession
        rt.sessions["s1"] = ElectronSession(session_id="s1", current_url="https://example.com/")
        return rt

    async def test_navigate_action(self, rt):
        nav_resp = {"ok": True, "url": "https://example.com/new", "title": "New page"}
        observe_resp = {**_OBSERVE_RESPONSE, "url": "https://example.com/new"}
        responses = iter([nav_resp, observe_resp])
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _mock_urlopen(next(responses))):
            result = await rt.action("s1", {"action": "navigate", "url": "https://example.com/new"})
        assert result["ok"] is True
        assert result["current_url"] == "https://example.com/new"

    async def test_click_action(self, rt):
        click_resp = {**_OBSERVE_RESPONSE, "ok": True}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(click_resp)):
            result = await rt.action("s1", {"action": "click", "element_id": "ae-1"})
        assert result["ok"] is True

    async def test_fill_action(self, rt):
        fill_resp = {**_OBSERVE_RESPONSE, "ok": True}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(fill_resp)):
            result = await rt.action("s1", {"action": "fill", "element_id": "ae-2", "text": "hello"})
        assert result["ok"] is True

    async def test_scroll_action(self, rt):
        scroll_resp = {**_OBSERVE_RESPONSE, "ok": True}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(scroll_resp)):
            result = await rt.action("s1", {"action": "scroll", "amount": 500})
        assert result["ok"] is True

    async def test_wait_action_returns_observe(self, rt):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_OBSERVE_RESPONSE)):
            result = await rt.action("s1", {"action": "wait", "milliseconds": 10})
        assert result["ok"] is True

    async def test_unsupported_action_returns_error(self, rt):
        result = await rt.action("s1", {"action": "hover"})
        assert result["ok"] is False
        assert "Unsupported" in result["error"]

    async def test_missing_action_key_returns_error(self, rt):
        result = await rt.action("s1", {})
        assert result["ok"] is False

    async def test_action_logs_are_recorded(self, rt):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({**_OBSERVE_RESPONSE, "ok": True})):
            await rt.action("s1", {"action": "scroll", "amount": 100})
        log = rt.sessions["s1"].action_log
        assert len(log) == 1
        assert log[0]["action"] == "scroll"
        assert log[0]["status"] == "executed"


class TestElectronBrowserRuntimeStatus:

    @pytest.fixture()
    def rt(self):
        rt = ElectronBrowserRuntime(port=47200)
        from theriad.browser.electron_runtime import ElectronSession
        rt.sessions["s1"] = ElectronSession(session_id="s1", current_url="https://example.com/")
        return rt

    async def test_status_known_session(self, rt):
        result = await rt.status("s1")
        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["backend"] == "electron"

    async def test_status_unknown_session(self, rt):
        result = await rt.status("ghost")
        assert result["ok"] is False
        assert result["status"] == "not_found"

    async def test_status_global_when_electron_running(self, rt):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_HEALTH_RESPONSE)):
            result = await rt.status()
        assert result["ok"] is True
        assert result["active_sessions"] == 1
        assert result["status"] == "available"
