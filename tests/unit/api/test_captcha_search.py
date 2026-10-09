"""CAPTCHA handoff and API-first search regression tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from theriad.agents.browser_events import ObservationEvent
from theriad.agents.browser_observation import detect_captcha
from theriad.agents.browser_runtime import run_until_pause
from theriad.agents.browser_schemas import BrowserAgentStartRequest, STATUS_NEEDS_APPROVAL
from theriad.agents.browser_session_store import BrowserSession
from theriad.search import SearchResult


def test_captcha_text_fixture_is_detected():
    detected, reason = detect_captcha(
        "https://www.google.com/search?q=test",
        "Before you continue",
        "Our systems have detected unusual traffic from your computer network. reCAPTCHA",
    )
    assert detected is True
    assert reason


def test_google_sorry_url_is_detected():
    detected, reason = detect_captcha(
        "https://www.google.com/sorry/index?continue=search",
        "Google",
        "",
    )
    assert detected is True
    assert "Google" in reason


@pytest.mark.asyncio
async def test_agent_stops_once_on_captcha_with_final_answer(monkeypatch):
    import theriad.agents.browser_runtime as runtime

    calls = 0

    async def captcha_observe(url, **kwargs):
        nonlocal calls
        calls += 1
        return ObservationEvent(
            url="https://www.google.com/sorry/index",
            title="Google unusual traffic",
            observation_type="dom",
            visible_text="I'm not a robot — reCAPTCHA",
            captcha_detected=True,
            captcha_reason="CAPTCHA marker detected: recaptcha",
        )

    monkeypatch.setattr(runtime, "observe_page", captcha_observe)
    session = BrowserSession(BrowserAgentStartRequest(
        goal="Research current browser automation news",
        start_url="https://www.google.com/search?q=browser+automation",
        max_steps=10,
    ))

    await run_until_pause(session, mock_mode=True)
    first_steps = session.steps_taken
    await run_until_pause(session, mock_mode=True)

    assert session.status == STATUS_NEEDS_APPROVAL
    assert session.captcha_detected is True
    assert session.final_answer
    assert "CAPTCHA encountered" in session.final_answer
    assert session.current_url.endswith("/sorry/index")
    assert session.page_title == "Google unusual traffic"
    assert session.steps_taken == first_steps
    assert calls == 1


def test_web_search_prompt_calls_search_service_before_browser(monkeypatch):
    import theriad.agents.browser_observation as observation_module
    from theriad.search.service import SearchService

    calls = []

    async def fake_search(self, query):
        calls.append(query)
        return [SearchResult(
            title="Bethesda listings",
            url="https://listings.example/bethesda",
            snippet="Example result",
            source="test-api",
        )]

    monkeypatch.setattr(SearchService, "search", fake_search)
    monkeypatch.setattr(
        observation_module, "_playwright_chromium_installed", lambda: False
    )

    from theriad.api.server import app
    with TestClient(app) as client:
        response = client.post("/api/chat/agent", json={
            "message": "Find cheap real estate listings in Bethesda MD",
            "max_steps": 2,
            "real_observation_enabled": False,
            "real_action_enabled": False,
        })

    assert response.status_code == 200
    data = response.json()
    assert calls == ["Find cheap real estate listings in Bethesda MD"]
    assert data["trace"]["search_results"][0]["source"] == "test-api"
    assert data["trace"]["search_results"][0]["url"].startswith(
        "https://listings.example/"
    )
    # Structured API results complete the lookup without opening a browser.
    assert data["browser_session_id"] is None
    assert data["trace"]["current_url"] is None
    assert "Bethesda listings" in data["trace"]["final_answer"]


def test_chat_search_fails_closed_when_api_is_unavailable(monkeypatch):
    from theriad.search.service import SearchService

    async def unavailable_search(self, query):
        raise ConnectionError("private upstream detail")

    monkeypatch.setattr(SearchService, "search", unavailable_search)

    from theriad.api.server import app
    with TestClient(app) as client:
        response = client.post("/api/chat/agent", json={
            "message": "Find food near me in Philly",
            "max_steps": 2,
            "real_observation_enabled": False,
            "real_action_enabled": False,
        })

    assert response.status_code == 200
    data = response.json()
    assert data["phase"] == "failed"
    assert data["browser_session_id"] is None
    assert data["trace"]["current_url"] is None
    assert "Automatic web search is unavailable" in data["trace"]["final_answer"]
    assert "private upstream detail" not in str(data)


def test_direct_browser_start_does_not_fall_back_when_opensandbox_is_unavailable(monkeypatch):
    import theriad.agents.browser_observation as observation_module
    from theriad.search.service import SearchService

    async def unavailable_search(self, query):
        raise ConnectionError("private upstream detail")

    monkeypatch.setattr(SearchService, "search", unavailable_search)
    monkeypatch.setenv("THERIAD_BROWSER_ACTION_BACKEND", "mock")
    monkeypatch.setenv("THERIAD_BROWSER_OBSERVATION_BACKEND", "mock")
    monkeypatch.setenv("THERIAD_ALLOW_MOCK_BROWSER", "true")
    monkeypatch.setattr(
        observation_module, "_playwright_chromium_installed", lambda: False
    )

    from theriad.api.server import app
    with TestClient(app) as client:
        response = client.post("/api/agent/browser/start", json={
            "goal": "Find food near me in Philly",
            "max_steps": 1,
            "real_observation_enabled": False,
            "real_action_enabled": False,
        })

    assert response.status_code == 503
    data = response.json()
    assert data["backend"] == "opensandbox"
    assert data["preferred_backend"] == "opensandbox"
    assert "opensandbox" in data["error"].lower()
    assert "private upstream detail" not in str(data)
