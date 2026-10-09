"""Initial browser navigation and placeholder rewrite tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from theriad.api.server import app
from theriad.search import (
    SearchResult,
    SearchService,
    is_unresolved_search_url,
    resolve_initial_navigation,
)


PROMPT = "Find me a decent house in the Philly area"


def test_housing_prompt_defaults_to_google(monkeypatch):
    monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
    result = resolve_initial_navigation(PROMPT)
    assert result.resolved_url.startswith("https://www.google.com/search?q=")
    assert "Philly" in result.resolved_url
    assert not is_unresolved_search_url(result.resolved_url)
    assert "example.com" not in result.resolved_url


def test_configured_search_provider(monkeypatch):
    monkeypatch.setenv("THERIAD_SEARCH_PROVIDER", "bing")
    result = resolve_initial_navigation(PROMPT)
    assert result.resolved_url.startswith("https://www.bing.com/search?q=")
    assert "example.com" not in result.resolved_url


def test_agent_needs_no_start_url_does_not_escalate_when_search_fails(monkeypatch):
    import theriad.agents.browser_observation as observation_module

    monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
    async def unavailable_search(_message, *, telemetry):
        telemetry["execution_kind"] = "web_search"
        raise ConnectionError("search unavailable")

    monkeypatch.setattr(
        "theriad.agents.search_synthesis.synthesize_web_search",
        unavailable_search,
    )
    monkeypatch.setattr(
        observation_module, "_playwright_chromium_installed", lambda: False
    )
    with TestClient(app) as client:
        response = client.post("/api/chat/agent", json={
            "message": PROMPT,
            "max_steps": 2,
            "real_observation_enabled": False,
            "real_action_enabled": False,
        })
    data = response.json()
    assert response.status_code == 200
    trace = data["trace"]
    if trace["search_results"]:
        # Fresh installs answer ordinary web questions directly through the
        # automatic search path; no browser launch page is needed.
        assert data["phase"] == "completed"
        assert trace["current_url"] is None
        assert trace["final_answer"]
    else:
        # If the canonical API search is unavailable, fail closed rather than
        # silently opening a search-engine page.
        assert data["phase"] == "failed"
        assert trace["current_url"] is None
        assert "Automatic web search is unavailable" in trace["final_answer"]


def test_placeholder_start_url_is_rejected_without_opensandbox(monkeypatch):
    import theriad.agents.browser_observation as observation_module

    monkeypatch.setenv("THERIAD_SEARCH_PROVIDER", "google")
    monkeypatch.setattr(
        observation_module, "_playwright_chromium_installed", lambda: False
    )
    with TestClient(app) as client:
        response = client.post("/api/agent/browser/start", json={
            "goal": PROMPT,
            "start_url": "https://example.com/?q=test",
            "max_steps": 1,
        })
    assert response.status_code == 503
    data = response.json()
    assert data["backend"] == "opensandbox"
    assert data["preferred_backend"] == "opensandbox"
    assert "current_url" not in data


def test_unresolved_google_homepage_start_url_is_rejected_without_opensandbox(monkeypatch):
    import theriad.agents.browser_observation as observation_module

    monkeypatch.setenv("THERIAD_SEARCH_PROVIDER", "google")
    monkeypatch.setattr(
        observation_module, "_playwright_chromium_installed", lambda: False
    )
    with TestClient(app) as client:
        response = client.post("/api/agent/browser/start", json={
            "goal": PROMPT,
            "start_url": "https://www.google.com/search",
            "max_steps": 1,
        })
    assert response.status_code == 503
    data = response.json()
    assert data["backend"] == "opensandbox"
    assert data["preferred_backend"] == "opensandbox"
    assert "current_url" not in data


def test_unresolved_search_url_detection():
    assert is_unresolved_search_url("https://www.google.com/search")
    assert is_unresolved_search_url("https://www.bing.com/search")
    assert is_unresolved_search_url("https://www.google.com/search?q=")
    assert not is_unresolved_search_url("https://www.google.com/search?q=philly")
    assert not is_unresolved_search_url("https://www.bing.com/search?q=philly")


class _HomepageProvider:
    name = "bad-provider"

    async def search(self, query: str) -> list[SearchResult]:
        return [
            SearchResult(
                title="Bare homepage",
                url="https://www.google.com/search",
                snippet="Should be rewritten before browser execution.",
                source=self.name,
            )
        ]


async def test_search_service_rewrites_homepage_provider_result():
    results = await SearchService(_HomepageProvider()).search(PROMPT)
    assert results[0].url.startswith("https://www.google.com/search?q=")
    assert "Philly" in results[0].url
