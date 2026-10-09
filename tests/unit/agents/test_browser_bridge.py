"""BrowserBridge foundation and runtime integration tests."""
from __future__ import annotations

import inspect

import pytest

from theriad.agents.browser_events import ObservationEvent
from theriad.agents.browser_runtime import run_step
from theriad.agents.browser_schemas import BrowserAgentStartRequest
from theriad.agents.browser_session_store import BrowserSession
from theriad.browser.bridge import BrowserBridge
from theriad.browser.policy import BrowserPolicy


async def _observer(url: str, *, use_screenshot=False, **_):
    return ObservationEvent(
        url=url,
        title="Test",
        observation_type="both" if use_screenshot else "dom",
        visible_text="page text",
        clickable_elements=[],
        screenshot_description="captured" if use_screenshot else None,
    )


@pytest.mark.asyncio
async def test_bridge_defaults_to_mock_backend():
    bridge = BrowserBridge(observer=_observer)
    started = await bridge.start_session({"start_url": "https://example.com"})
    assert started.ok
    assert started.backend == "mock"
    result = await bridge.read_page(started.session_id)
    assert result.ok
    assert result.backend == "mock"
    assert result.url == "https://example.com"


def test_bridge_selects_playwright_backend():
    bridge = BrowserBridge(backend="playwright", observer=_observer)
    assert bridge.backend == "playwright"


def test_playwright_adapter_never_imports_sync_playwright():
    import theriad.browser.adapters as adapters

    source = inspect.getsource(adapters.PlaywrightBrowserAdapter)
    assert "sync_playwright" not in source
    assert "playwright.sync_api" not in source


@pytest.mark.asyncio
async def test_playwright_unavailable_is_clear():
    bridge = BrowserBridge(backend="playwright", observer=_observer)
    bridge.adapter.available = False
    bridge.adapter.unavailable_error = "Playwright is unavailable."
    result = await bridge.start_session()
    assert not result.ok
    assert result.status == "failed"
    assert "Playwright is unavailable" in result.error


def test_browser_policy_classification():
    policy = BrowserPolicy()
    assert policy.classify("navigate").status == "success"
    for action in ("submit_form", "login", "payment"):
        assert policy.classify(action).status == "approval_required"
    for action in ("bypass_captcha", "credential_theft", "exfiltrate_secrets"):
        assert policy.classify(action).status == "blocked"


@pytest.mark.asyncio
async def test_runtime_observe_and_navigate_use_bridge(monkeypatch):
    import theriad.agents.browser_runtime as runtime

    calls = {"observe": 0, "navigate": 0}

    async def fake_observe(url, **kwargs):
        calls["observe"] += 1
        return ObservationEvent(
            url=url,
            title="",
            observation_type="dom",
            visible_text="",
            clickable_elements=[],
        )

    original_navigate = BrowserBridge.navigate

    async def spy_navigate(self, session_id, url):
        calls["navigate"] += 1
        return await original_navigate(self, session_id, url)

    monkeypatch.setattr(runtime, "observe_page", fake_observe)
    monkeypatch.setattr(BrowserBridge, "navigate", spy_navigate)
    session = BrowserSession(BrowserAgentStartRequest(
        goal="Go to wikipedia and summarize AI",
        max_steps=3,
    ))
    session.real_action_enabled = True
    session.action_backend = "playwright"
    session.observation_backend = "playwright"

    await run_step(session, mock_mode=False)

    assert calls["observe"] >= 1
    assert calls["navigate"] == 1
