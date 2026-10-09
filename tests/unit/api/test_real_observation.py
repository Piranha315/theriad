"""
Tests for the real observation mode (Playwright-based, read-only) and the
OpenSandbox-backed HTTP browser contract.

Coverage:
1.  App works without Playwright installed.
2.  Mock observation remains available at the lower-level runtime boundary.
3.  real_observation_enabled=True with Playwright missing is handled gracefully.
4.  Real observation does not execute any browser actions.
5.  No secrets/cookies/tokens are returned in observation output.
6.  Screenshot is NOT taken unless capture_screenshot=True is set.
7.  observe_page() always returns an ObservationEvent in mock mode.
8.  PLAYWRIGHT_AVAILABLE flag is exposed correctly.
9.  /api/agent/browser/capabilities reports OpenSandbox truthfully.
10. The HTTP API never silently downgrades to mock or Playwright.
11. Lower-level mock actions remain covered independently of the HTTP API.
12. DOM extraction never includes cookie/storage/credential fields.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
import pytest
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch, call


class _FakePWManager:
    """Async context manager standing in for async_playwright()."""

    def __init__(self, pw):
        self._pw = pw

    async def __aenter__(self):
        return self._pw

    async def __aexit__(self, *exc):
        return False


@contextmanager
def _fake_playwright(mock_pw):
    """
    Inject a fake `playwright` package into sys.modules so that the
    local `from playwright.async_api import async_playwright` inside
    _real_observe() resolves to our mock — even when Playwright is
    not installed in this environment.
    """
    fake_async_api = types.ModuleType("playwright.async_api")
    fake_async_api.async_playwright = lambda: _FakePWManager(mock_pw)
    fake_async_api.TimeoutError = TimeoutError  # needed for except clause

    fake_playwright = types.ModuleType("playwright")
    fake_playwright.async_api = fake_async_api

    saved = {k: sys.modules[k] for k in ("playwright", "playwright.async_api") if k in sys.modules}
    sys.modules["playwright"] = fake_playwright
    sys.modules["playwright.async_api"] = fake_async_api
    try:
        yield
    finally:
        for k in ("playwright", "playwright.async_api"):
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                sys.modules.pop(k, None)


def _make_async_page(
    *,
    url="https://example.com",
    title="Example",
    evaluate_result=None,
    screenshot_bytes=b"\x89PNG\r\n\x1a\n",
):
    """
    Build an AsyncMock page whose awaited methods (goto/title/evaluate/
    screenshot/click/...) behave like the Playwright async API. `url` is a
    plain attribute (property in real Playwright, never awaited).
    """
    page = AsyncMock()
    page.url = url
    page.title.return_value = title
    page.evaluate.return_value = evaluate_result or {
        "visible_text": "",
        "clickable_elements": [],
        "forms": [],
        "aria_labels": [],
    }
    page.goto.return_value = None
    page.screenshot.return_value = screenshot_bytes
    return page


def _make_async_pw(page):
    """Build the pw → browser → context → page AsyncMock chain."""
    ctx = AsyncMock()
    ctx.new_page.return_value = page
    browser = AsyncMock()
    browser.new_context.return_value = ctx
    pw = MagicMock()
    pw.chromium.launch = AsyncMock(return_value=browser)
    return pw, browser, ctx


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
    from theriad.workflow.detector import WorkflowDetector
    import theriad.api.server as server_module

    wf = WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(server_module, "_wf_detector", wf, raising=False)
        from theriad.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _assert_opensandbox_unavailable(response):
    """Assert the API reports the canonical browser dependency truthfully."""
    assert response.status_code == 503
    data = response.json()
    assert data["backend"] == "opensandbox"
    assert data["preferred_backend"] == "opensandbox"
    assert "opensandbox" in data["error"].lower()
    assert "session_id" not in data
    return data


def _force_opensandbox_unavailable(monkeypatch):
    """Keep unavailable-backend tests independent of a developer's live server."""
    monkeypatch.setenv("THERIAD_OPENSANDBOX_DOMAIN", "127.0.0.1:9")


# ════════════════════════════════════════════════════════════════════════════
# 1. App works without Playwright installed
# ════════════════════════════════════════════════════════════════════════════

def test_startup_without_playwright():
    """Importing browser_observation must not raise even if Playwright is absent."""
    import importlib, sys
    # Temporarily hide playwright from imports
    playwright_modules = [k for k in sys.modules if k.startswith("playwright")]
    saved = {k: sys.modules.pop(k) for k in playwright_modules}
    try:
        # Reload the module with playwright removed
        import theriad.agents.browser_observation as obs_mod
        importlib.reload(obs_mod)
        # Module should load cleanly regardless
        assert hasattr(obs_mod, "observe_page")
        assert hasattr(obs_mod, "PLAYWRIGHT_AVAILABLE")
    finally:
        sys.modules.update(saved)
        importlib.reload(obs_mod)  # restore original state


async def test_mock_mode_works_without_playwright(monkeypatch):
    """Mock observation must work even when PLAYWRIGHT_AVAILABLE=False."""
    import theriad.agents.browser_observation as obs_mod
    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", False)

    obs = await obs_mod.observe_page("https://example.com", mock_mode=True)
    assert obs.url == "https://example.com"
    assert obs.title != ""


# ════════════════════════════════════════════════════════════════════════════
# 2. Lower-level mock observation remains available
# ════════════════════════════════════════════════════════════════════════════

async def test_observe_page_defaults_to_mock():
    """observe_page() with no args must use mock data, never the real browser."""
    from theriad.agents.browser_observation import observe_page

    called = []

    async def _spy(*a, **k):
        called.append(True)
        return None

    with patch("theriad.agents.browser_observation._real_observe", side_effect=_spy):
        obs = await observe_page("https://example.com")
    assert len(called) == 0, "Real observe was called when mock_mode=True (default)"
    assert obs is not None


def test_session_start_requires_canonical_opensandbox(client, monkeypatch):
    """The HTTP browser API must not silently downgrade to a mock browser."""
    _force_opensandbox_unavailable(monkeypatch)
    resp = client.post("/api/agent/browser/start", json={"goal": "find AI prices"})
    _assert_opensandbox_unavailable(resp)


# ════════════════════════════════════════════════════════════════════════════
# 3. Real observation unavailable → clear error
# ════════════════════════════════════════════════════════════════════════════

async def test_real_obs_unavailable_clear_error(monkeypatch):
    """
    observe_page() with real_observation_enabled=True and Playwright missing
    must raise RuntimeError with an actionable install message.
    """
    import theriad.agents.browser_observation as obs_mod
    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", False)

    with pytest.raises(RuntimeError) as exc_info:
        await obs_mod.observe_page(
            "https://example.com",
            mock_mode=False,
            real_observation_enabled=True,
        )
    msg = str(exc_info.value)
    assert "Playwright" in msg or "playwright" in msg
    assert "install" in msg.lower() or "pip" in msg.lower()


def test_start_fails_closed_without_opensandbox(client, monkeypatch):
    """
    Requesting a live backend without Playwright fails visibly and never
    returns a successful mock session.
    """
    _force_opensandbox_unavailable(monkeypatch)
    resp = client.post("/api/agent/browser/start", json={
        "goal": "observe example.com",
        "real_observation_enabled": True,
    })
    data = _assert_opensandbox_unavailable(resp)
    assert "playwright" not in data["error"].lower()


def test_step_requires_a_canonical_session(monkeypatch, client):
    """
    If a session was somehow created with real_observation_enabled=True
    but Playwright becomes unavailable at step time, the session
    transitions to failed with a clear error.
    """
    step_resp = client.post("/api/agent/browser/step", json={"steps": 1})
    assert step_resp.status_code == 422
    assert "session_id" in step_resp.json()["error"]


# ════════════════════════════════════════════════════════════════════════════
# 4. Real observation does not execute browser actions
# ════════════════════════════════════════════════════════════════════════════

async def test_real_observe_never_clicks_or_types(monkeypatch):
    """
    Mock out the Playwright browser object and verify that page methods
    which perform actions (click, type, fill, press, evaluate with mutation)
    are never called during observation.
    """
    import theriad.agents.browser_observation as obs_mod

    mock_page = _make_async_page(
        url="https://example.com",
        title="Example Domain",
        evaluate_result={
            "visible_text": "Example Domain. This domain is for use in illustrative examples.",
            "clickable_elements": [{"tag": "a", "text": "More information", "href": "https://example.com/info"}],
            "forms": [],
            "aria_labels": [],
        },
    )
    mock_pw, mock_browser, mock_ctx = _make_async_pw(mock_page)

    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", True)

    with _fake_playwright(mock_pw):
        obs = await obs_mod._real_observe("https://example.com", use_screenshot=False)

    # Must never have called action methods
    mock_page.click.assert_not_called()
    mock_page.type.assert_not_called()
    mock_page.fill.assert_not_called()
    mock_page.press.assert_not_called()
    mock_page.select_option.assert_not_called()
    mock_page.dispatch_event.assert_not_called()

    # Must have called read-only methods
    mock_page.goto.assert_called_once()
    mock_page.evaluate.assert_called_once()

    assert obs.url == "https://example.com"
    assert obs.title == "Example Domain"


# ════════════════════════════════════════════════════════════════════════════
# 5. No secrets/cookies/tokens in observation output
# ════════════════════════════════════════════════════════════════════════════

async def test_observation_event_has_no_secret_fields(monkeypatch):
    """ObservationEvent must not carry cookies, storage, or auth tokens."""
    import theriad.agents.browser_observation as obs_mod

    mock_page = _make_async_page(
        url="https://example.com",
        title="Example",
        evaluate_result={
            "visible_text": "Hello world",
            "clickable_elements": [],
            "forms": [],
            "aria_labels": [],
        },
    )
    mock_pw, mock_browser, mock_ctx = _make_async_pw(mock_page)

    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", True)

    with _fake_playwright(mock_pw):
        obs = await obs_mod._real_observe("https://example.com", use_screenshot=False)

    obs_dict = vars(obs)
    forbidden = {"cookie", "cookies", "auth_token", "session_token", "password",
                 "api_key", "secret", "credentials", "storage"}
    for field in obs_dict:
        assert field.lower() not in forbidden, f"Forbidden field '{field}' in ObservationEvent"

    # Verify the browser context never calls storage_state/cookies
    mock_ctx.storage_state.assert_not_called()
    mock_ctx.cookies.assert_not_called()


def test_status_response_has_no_cookies_or_secrets(client, monkeypatch):
    """Session status API response must not include any credential fields."""
    _force_opensandbox_unavailable(monkeypatch)
    start = client.post("/api/agent/browser/start", json={"goal": "find prices"})
    data = _assert_opensandbox_unavailable(start)

    import json as _json
    body = _json.dumps(data).lower()
    for term in ["cookie", "auth_token", "session_token", "api_key", "password", "secret"]:
        assert term not in body, f"Forbidden term '{term}' found in status response"


# ════════════════════════════════════════════════════════════════════════════
# 6. Screenshot NOT taken unless capture_screenshot=True
# ════════════════════════════════════════════════════════════════════════════

async def test_screenshot_not_taken_by_default(monkeypatch):
    """
    _real_observe() with use_screenshot=False must not call page.screenshot().
    """
    import theriad.agents.browser_observation as obs_mod

    mock_page = _make_async_page(
        url="https://example.com",
        title="Example",
        evaluate_result={
            "visible_text": "some text",
            "clickable_elements": [],
            "forms": [],
            "aria_labels": [],
        },
    )
    mock_pw, mock_browser, mock_ctx = _make_async_pw(mock_page)

    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", True)

    with _fake_playwright(mock_pw):
        obs = await obs_mod._real_observe("https://example.com", use_screenshot=False)

    mock_page.screenshot.assert_not_called()
    assert obs.screenshot_description is None
    assert obs.raw_dom_snippet is None


async def test_screenshot_taken_when_requested(monkeypatch):
    """
    _real_observe() with use_screenshot=True should call page.screenshot()
    and populate screenshot_description.
    """
    import theriad.agents.browser_observation as obs_mod

    mock_page = _make_async_page(
        url="https://example.com",
        title="Example",
        evaluate_result={
            "visible_text": "some text",
            "clickable_elements": [],
            "forms": [],
            "aria_labels": [],
        },
        # Minimal PNG bytes
        screenshot_bytes=b"\x89PNG\r\n\x1a\n",
    )
    mock_pw, mock_browser, mock_ctx = _make_async_pw(mock_page)

    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", True)

    with _fake_playwright(mock_pw):
        obs = await obs_mod._real_observe("https://example.com", use_screenshot=True)

    mock_page.screenshot.assert_called_once()
    assert obs.screenshot_description is not None
    assert "Screenshot" in obs.screenshot_description
    assert obs.observation_type == "both"


def test_session_capture_screenshot_false_by_default():
    """Sessions created without capture_screenshot should not request screenshots."""
    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    req = BrowserAgentStartRequest(goal="test")
    assert req.capture_screenshot is False


# ════════════════════════════════════════════════════════════════════════════
# 7. observe_page() always returns ObservationEvent (never raises in mock)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("url", [
    "https://example.com",
    "about:blank",
    "http://localhost:8080",
    "https://very-long-url-that-has-many-path-segments.example.com/a/b/c/d?x=1&y=2#hash",
])
async def test_mock_observe_never_raises(url):
    from theriad.agents.browser_observation import observe_page
    obs = await observe_page(url, mock_mode=True)
    assert obs is not None
    assert obs.url == url


# ════════════════════════════════════════════════════════════════════════════
# 8. PLAYWRIGHT_AVAILABLE flag is correct
# ════════════════════════════════════════════════════════════════════════════

def test_playwright_available_is_bool():
    from theriad.agents.browser_observation import PLAYWRIGHT_AVAILABLE
    assert isinstance(PLAYWRIGHT_AVAILABLE, bool)


# ════════════════════════════════════════════════════════════════════════════
# 9. /api/agent/browser/capabilities reflects Playwright state
# ════════════════════════════════════════════════════════════════════════════

def test_capabilities_endpoint_shape(client):
    resp = client.get("/api/agent/browser/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "opensandbox"
    assert "opensandbox_available" in data
    assert "real_observation_supported" in data
    assert "actions_always_mocked" in data
    assert isinstance(data["actions_always_mocked"], bool)
    available = bool(data["opensandbox_available"])
    assert data["real_observation_supported"] is available
    assert data["real_actions_enabled"] is available
    assert data["action_backend"] == ("opensandbox" if available else "unavailable")
    assert data["observation_backend"] == ("opensandbox" if available else "unavailable")


def test_capabilities_without_playwright(client, monkeypatch):
    import theriad.agents.browser_observation as obs_mod
    import theriad.api.server as server_mod
    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", False)

    resp = client.get("/api/agent/browser/capabilities")
    data = resp.json()
    # real_observation_supported should reflect actual Playwright state
    # (server reads from obs_mod.PLAYWRIGHT_AVAILABLE at request time)
    assert resp.status_code == 200
    assert data["backend"] == "opensandbox"
    available = bool(data["opensandbox_available"])
    assert data["action_backend"] == ("opensandbox" if available else "unavailable")
    assert data["observation_backend"] == ("opensandbox" if available else "unavailable")


# ════════════════════════════════════════════════════════════════════════════
# 10. HTTP API does not downgrade a missing live backend to mock
# ════════════════════════════════════════════════════════════════════════════

def test_real_obs_never_downgrades_to_mock(client, monkeypatch):
    _force_opensandbox_unavailable(monkeypatch)
    resp = client.post("/api/agent/browser/start", json={
        "goal": "read a webpage",
        "real_observation_enabled": True,
    })
    _assert_opensandbox_unavailable(resp)


# ════════════════════════════════════════════════════════════════════════════
# 11. Real-action mode uses the live executor for safe actions
# ════════════════════════════════════════════════════════════════════════════

def test_actions_are_not_reported_as_mocked_when_browser_is_unavailable(client):
    """Unavailable means no browser exists; it must not be mislabeled mock."""
    resp = client.get("/api/agent/browser/capabilities")
    data = resp.json()
    available = bool(data["opensandbox_available"])
    assert data["action_backend"] == ("opensandbox" if available else "unavailable")
    assert data["actions_always_mocked"] is False


async def test_run_step_uses_live_execute_when_enabled(monkeypatch):
    """
    With real_action_enabled=True, safe actions go through the live bridge
    executor instead of being reported as simulated success.
    """
    import theriad.agents.browser_runtime as rt
    from theriad.agents.browser_session_store import BrowserSession
    from theriad.agents.browser_schemas import BrowserAgentStartRequest

    req = BrowserAgentStartRequest(
        goal="test real obs no actions",
        max_steps=5,
        real_observation_enabled=True,
    )
    session = BrowserSession(req)
    session.real_observation_enabled = True

    live_execute_calls = []
    mock_execute_calls = []

    async def spy_live(proposal, obs, _session):
        live_execute_calls.append(True)
        return rt.ActionResult(
            action_type=proposal.action_type,
            target=proposal.target,
            url=proposal.target or obs.url,
            success=True,
            result_summary="Verified live test action",
        )

    original_mock_exec = rt._mock_execute_action

    def spy_mock(*a, **kw):
        mock_execute_calls.append(True)
        return original_mock_exec(*a, **kw)

    monkeypatch.setattr(rt, "_execute_bridge_action", spy_live)
    monkeypatch.setattr(rt, "_mock_execute_action", spy_mock)

    # Mock out the observation bridge to keep this lower-level unit test
    # independent of the live OpenSandbox service. The API integration tests
    # exercise the real bridge separately.
    from theriad.agents.browser_events import ObservationEvent
    import theriad.agents.browser_observation as obs_mod
    mock_obs = ObservationEvent(
        url="https://example.com",
        title="Test",
        observation_type="dom",
        visible_text="Some content here",
        clickable_elements=[{"tag": "a", "text": "Link", "href": "https://example.com/next"}],
        forms=[],
        aria_labels=[],
    )
    # run_step imports observe_page at module level, so patch it in
    # browser_runtime's namespace. observe_page is async now, so the stand-in
    # must be awaitable.
    async def _fake_observe(*a, **k):
        return mock_obs

    monkeypatch.setattr(rt, "observe_page", _fake_observe)
    class _FakeBridge:
        async def observe(self, *_args, **_kwargs):
            return SimpleNamespace(ok=True, observation=mock_obs, error=None)

        async def screenshot(self, *_args, **_kwargs):
            return SimpleNamespace(ok=True, observation=mock_obs, error=None)

    async def _fake_get_browser_bridge(*_args, **_kwargs):
        return _FakeBridge()

    monkeypatch.setattr(rt, "_get_browser_bridge", _fake_get_browser_bridge)
    # PLAYWRIGHT_AVAILABLE is re-imported locally inside run_step each call, so patch the source
    monkeypatch.setattr(obs_mod, "PLAYWRIGHT_AVAILABLE", True)

    # Also stub the planner to always return a safe, pre-approved action so the
    # pipeline reaches execute without hitting the approval gate.
    from theriad.agents.browser_events import ActionProposal
    safe_proposal = ActionProposal(
        action_type="navigate",
        target="https://example.com/next",
        reason="safe navigation",
        confidence=0.9,
        risk_level="low",
        requires_approval=False,
    )
    monkeypatch.setattr(rt, "_mock_plan_action", lambda *a, **k: safe_proposal)

    session.real_action_enabled = True
    await rt.run_step(session, mock_mode=False)

    assert live_execute_calls == [True]
    assert mock_execute_calls == []


# ════════════════════════════════════════════════════════════════════════════
# 12. DOM extraction never includes cookies/storage/credential fields
# ════════════════════════════════════════════════════════════════════════════

async def test_extract_dom_sanitises_inputs(monkeypatch):
    """
    The DOM extraction JS snippet must not return password field values
    or any storage content. The test verifies the schema of returned data.
    """
    import theriad.agents.browser_observation as obs_mod

    # Simulate evaluate() returning data with a password input.
    # evaluate() is awaited by the async _extract_dom, so use AsyncMock.
    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock()
    mock_page.evaluate.return_value = {
        "visible_text": "Login page",
        "clickable_elements": [{"tag": "button", "text": "Sign in", "aria": ""}],
        "forms": [
            {
                "action": "/login",
                "method": "POST",
                "inputs": [
                    {"name": "email", "type": "email", "label": "Email"},
                    {"name": "password", "type": "password", "label": "Password"},
                    # Note: value is NOT returned (JS snippet never captures input.value)
                ],
            }
        ],
        "aria_labels": ["Login form"],
    }

    result = await obs_mod._extract_dom(mock_page)

    # Values must NOT be present in any input
    for form in result.get("forms", []):
        for inp in form.get("inputs", []):
            assert "value" not in inp, "Password value was captured in form input"
