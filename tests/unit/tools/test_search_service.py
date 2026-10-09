"""Automatic, zero-configuration web-search behavior."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from theriad.search.service import (
    BrowserSearchProvider,
    DdgHtmlSearchProvider,
    SearchResult,
    SearchService,
    SearXNGSearchProvider,
    _DdgHtmlResultParser,
    _decode_ddg_result_url,
    is_api_search_request,
    is_web_search_request,
)


@pytest.fixture(autouse=True)
def _reset_search_circuits():
    from theriad.runtime.circuit_breaker import get_circuit_registry

    registry = get_circuit_registry()
    registry.reset()
    yield
    registry.reset()


@pytest.mark.parametrize("profile_mode", ["ephemeral", "real_profile"])
def test_browser_preferences_do_not_disable_api_search(monkeypatch, profile_mode):
    monkeypatch.setenv("THERIAD_SEARCH_PROVIDER", "google")
    monkeypatch.setenv("THERIAD_CHROME_PROFILE_MODE", profile_mode)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("SCRAPLING_ENABLED", "false")

    service = SearchService()

    assert isinstance(service.provider, DdgHtmlSearchProvider)
    assert not isinstance(service.provider, BrowserSearchProvider)


@pytest.mark.parametrize("prompt", [
    "Find today's weather",
    "What's the weather today in Philadelphia?",
    "Search online for the latest AI news",
    "What are the current prices for flights to Tokyo?",
    # No explicit search words — intent classifier decides; API search is the default
    "Who won last night?",
    "What is the population of Japan?",
    "Tell me about the latest developments in fusion energy",
])
def test_read_only_web_questions_use_api_search(prompt):
    assert is_api_search_request(prompt)


@pytest.mark.parametrize("prompt", [
    "Go to example.com and summarize it",
    "Browse example.com for the latest news",
    "Summarize this site: example.com and list the latest updates",
    "Log into my bank and download my statement",
    "Book the cheapest flight I can find",
    "Click the sign-in button",
    "Navigate to the settings page and submit the form",
])
def test_interactive_browser_actions_do_not_use_api_search(prompt):
    assert not is_api_search_request(prompt)


def test_browser_search_fallback_setting_is_deprecated(monkeypatch):
    monkeypatch.setenv("THERIAD_SEARCH_FALLBACK_TO_BROWSER", "true")

    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert SearchService._browser_fallback_enabled() is True


def test_ddg_html_parser_extracts_real_result_links():
    parser = _DdgHtmlResultParser()
    parser.feed(
        """
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage">Example page</a>
          <a class="result__snippet">A useful snippet.</a>
        </div>
        """
    )

    assert parser.records == [{
        "title": "Example page",
        "url": "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage",
        "snippet": "A useful snippet.",
    }]
    assert _decode_ddg_result_url(parser.records[0]["url"]) == "https://example.org/page"




def test_browser_preferences_do_not_disable_api_search(monkeypatch):
    monkeypatch.setenv("THERIAD_SEARCH_PROVIDER", "google")
    monkeypatch.setenv("THERIAD_CHROME_PROFILE_MODE", "real_profile")
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("SCRAPLING_ENABLED", "false")

    service = SearchService()

    assert isinstance(service.provider, DdgHtmlSearchProvider)
    assert not isinstance(service.provider, BrowserSearchProvider)


@pytest.mark.asyncio
async def test_canonical_searxng_provider_parses_structured_results():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{
                "title": "Python asyncio docs",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "content": "Concurrent code with async and await.",
            }]}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            assert url == "http://search.internal/search"
            assert kwargs["params"]["format"] == "json"
            return Response()

    with patch("theriad.search.service.httpx.AsyncClient", return_value=Client()):
        rows = await SearXNGSearchProvider("http://search.internal").search("python async")

    assert rows == [SearchResult(
        "Python asyncio docs",
        "https://docs.python.org/3/library/asyncio.html",
        "Concurrent code with async and await.",
        "searxng",
    )]


@pytest.mark.asyncio
async def test_unreachable_searxng_uses_automatic_keyless_fallback(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SCRAPLING_ENABLED", raising=False)
    monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("THERIAD_CHROME_PROFILE_MODE", raising=False)

    service = SearchService()
    assert isinstance(service.provider, SearXNGSearchProvider)

    async def fallback(self, query):
        return [SearchResult("Live result", "https://example.org", "Snippet", "duckduckgo")]

    with patch.object(DdgHtmlSearchProvider, "search", fallback):
        results = await service.search("current information")

    assert results[0].source == "duckduckgo"
    assert results[0].url == "https://example.org"


@pytest.mark.asyncio
async def test_unresolved_primary_results_use_keyless_fallback(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://search.internal")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SCRAPLING_ENABLED", raising=False)
    monkeypatch.delenv("THERIAD_SEARCH_FALLBACK_TO_BROWSER", raising=False)

    async def stale_primary(_self, _query):
        return [SearchResult(
            "Search homepage",
            "https://www.google.com/search",
            "",
            "searxng",
        )]

    async def keyless_fallback(_self, _query):
        return [SearchResult(
            "Actual result",
            "https://example.org/article",
            "A usable result page.",
            "duckduckgo",
        )]

    with patch.object(SearXNGSearchProvider, "search", stale_primary), patch.object(
        DdgHtmlSearchProvider, "search", keyless_fallback
    ):
        results = await SearchService().search("current information")

    assert [result.url for result in results] == ["https://example.org/article"]
    assert results[0].source == "duckduckgo"


@pytest.mark.asyncio
async def test_search_answer_falls_back_to_live_links_without_model(monkeypatch):
    from theriad.agents.search_synthesis import synthesize_web_search

    class FakeService:
        async def search(self, query):
            return [SearchResult(
                "Live result",
                "https://example.org",
                "Current evidence.",
                "duckduckgo",
            )]

    async def no_model(*args, **kwargs):
        raise RuntimeError("model unavailable")

    with patch("theriad.agents.llm_planner._call_model", new=no_model):
        answer, results, queries = await synthesize_web_search(
            "Find current information", service=FakeService()
        )

    assert queries == ["Find current information"]
    assert results[0]["url"] == "https://example.org"
    assert "I found these live web results" in answer
    assert "https://example.org" in answer
    assert "Ask a follow-up" not in answer
