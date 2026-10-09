"""
Unit tests for theriad.search.scrapling_provider.

All HTTP / browser I/O is mocked — no live network required.
scrapling itself is also mocked so the tests run without the optional dep.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build a minimal scrapling stub so tests run even without the real package.
# ---------------------------------------------------------------------------

def _make_scrapling_stub():
    scrapling = types.ModuleType("scrapling")
    fetchers = types.ModuleType("scrapling.fetchers")

    class _FakePage:
        def __init__(self, elements):
            self._elements = elements

        def css(self, selector):
            return self._elements

    class _FakeEl:
        def __init__(self, text="", attrib=None, children=None):
            self.text = text
            self.attrib = attrib or {}
            self._children = children or {}

        def css_first(self, selector):
            return self._children.get(selector)

    class Fetcher:
        def __init__(self, auto_match=True):
            pass

        def get(self, url, **kwargs):
            return _FakePage([])

    fetchers.Fetcher = Fetcher
    scrapling.fetchers = fetchers
    sys.modules["scrapling"] = scrapling
    sys.modules["scrapling.fetchers"] = fetchers

    return scrapling, fetchers, _FakePage, _FakeEl, Fetcher


_scrapling, _fetchers, _FakePage, _FakeEl, _Fetcher = _make_scrapling_stub()


# ---------------------------------------------------------------------------
# Import the module under test AFTER the stub is registered.
# ---------------------------------------------------------------------------

from theriad.search.scrapling_provider import ScraplingSearchProvider, _decode_ddg_href


# ---------------------------------------------------------------------------
# _decode_ddg_href
# ---------------------------------------------------------------------------

class TestDecodeDdgHref:
    def test_empty(self):
        assert _decode_ddg_href("") == ""

    def test_plain_http(self):
        assert _decode_ddg_href("https://example.com") == "https://example.com"

    def test_protocol_relative_non_ddg(self):
        assert _decode_ddg_href("//example.com/page") == "https://example.com/page"

    def test_ddg_redirect(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&rut=abc"
        assert _decode_ddg_href(href) == "https://www.python.org/"

    def test_ddg_redirect_https_prefix(self):
        href = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2F"
        assert _decode_ddg_href(href) == "https://github.com/"

    def test_ddg_missing_uddg(self):
        href = "//duckduckgo.com/l/?rut=abc"
        assert _decode_ddg_href(href) == ""

    def test_non_http_non_ddg(self):
        assert _decode_ddg_href("ftp://old.example.com") == ""


# ---------------------------------------------------------------------------
# ScraplingSearchProvider.is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_when_stub_registered(self):
        assert ScraplingSearchProvider.is_available() is True

    def test_unavailable_when_import_fails(self):
        with patch.dict(sys.modules, {"scrapling": None}):
            assert ScraplingSearchProvider.is_available() is False


# ---------------------------------------------------------------------------
# ScraplingSearchProvider.from_environment
# ---------------------------------------------------------------------------

class TestFromEnvironment:
    def test_stealth_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SCRAPLING_STEALTH", raising=False)
        p = ScraplingSearchProvider.from_environment()
        assert p._stealth is False

    def test_stealth_on_via_env(self, monkeypatch):
        monkeypatch.setenv("SCRAPLING_STEALTH", "1")
        p = ScraplingSearchProvider.from_environment()
        assert p._stealth is True

    def test_stealth_on_via_true(self, monkeypatch):
        monkeypatch.setenv("SCRAPLING_STEALTH", "true")
        p = ScraplingSearchProvider.from_environment()
        assert p._stealth is True


# ---------------------------------------------------------------------------
# ScraplingSearchProvider._search_sync
# ---------------------------------------------------------------------------

def _make_result_el(title: str, href: str, snippet: str) -> _FakeEl:
    title_el = _FakeEl(text=title, attrib={"href": href})
    snippet_el = _FakeEl(text=snippet)
    container = _FakeEl(
        children={
            "a.result__a": title_el,
            "a.result__snippet": snippet_el,
        }
    )
    return container


class TestSearchSync:
    def test_page_html_supports_current_response_body(self):
        response = types.SimpleNamespace(body=b"<html><title>Clean</title></html>")
        assert ScraplingSearchProvider._page_html(response) == (
            "<html><title>Clean</title></html>"
        )

    def test_first_css_supports_current_selector_api(self):
        class SelectorOnly:
            def css(self, selector):
                return [selector]

        assert ScraplingSearchProvider._first_css(SelectorOnly(), "a.result__a") == "a.result__a"

    def test_returns_parsed_results(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com"
        el = _make_result_el("Example", href, "A snippet")
        page = _FakePage([el])

        provider = ScraplingSearchProvider(stealth=False)
        with patch.object(provider, "_fetch_basic", return_value=page):
            results = provider._search_sync("test query")

        assert len(results) == 1
        assert results[0].title == "Example"
        assert results[0].url == "https://example.com"
        assert results[0].snippet == "A snippet"
        assert results[0].source == "scrapling"

    def test_skips_elements_without_title_link(self):
        el = _FakeEl(children={})  # no a.result__a
        page = _FakePage([el])

        provider = ScraplingSearchProvider(stealth=False)
        with patch.object(provider, "_fetch_basic", return_value=page):
            results = provider._search_sync("test")

        assert results == []

    def test_skips_non_http_urls(self):
        el = _make_result_el("Bad", "ftp://bad.example.com", "")
        page = _FakePage([el])

        provider = ScraplingSearchProvider(stealth=False)
        with patch.object(provider, "_fetch_basic", return_value=page):
            results = provider._search_sync("test")

        assert results == []

    def test_caps_at_max_results(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com"
        elements = [_make_result_el(f"Title {i}", href, "") for i in range(20)]
        page = _FakePage(elements)

        provider = ScraplingSearchProvider(stealth=False)
        with patch.object(provider, "_fetch_basic", return_value=page):
            results = provider._search_sync("test")

        assert len(results) == 8

    def test_returns_empty_when_fetch_is_none(self):
        provider = ScraplingSearchProvider(stealth=False)
        with patch.object(provider, "_fetch_basic", return_value=None):
            results = provider._search_sync("test")
        assert results == []

    def test_uses_stealth_fetch_when_stealth_true(self):
        page = _FakePage([])
        provider = ScraplingSearchProvider(stealth=True)
        with patch.object(provider, "_fetch_stealth", return_value=page) as mock_stealth:
            provider._search_sync("test")
        mock_stealth.assert_called_once()

    def test_basic_challenge_escalates_without_configuration(self):
        captcha_page = MagicMock()
        captcha_page.html = "<html><body>Just a moment... checking your browser</body></html>"
        clean_page = _FakePage([_make_result_el(
            "Clean result",
            "https://example.com/clean",
            "Returned after automatic protection handling",
        )])
        provider = ScraplingSearchProvider(stealth=False)
        with (
            patch.object(provider, "_fetch_basic", return_value=captcha_page),
            patch.object(provider, "_fetch_stealth", return_value=clean_page) as mock_stealth,
        ):
            results = provider._search_sync("current information")

        mock_stealth.assert_called_once()
        assert results[0].url == "https://example.com/clean"

    def test_rate_limited_basic_response_escalates_without_configuration(self):
        limited_page = MagicMock()
        limited_page.status = 202
        limited_page.html = "<html><body>temporarily limited</body></html>"
        clean_page = _FakePage([_make_result_el(
            "Clean result",
            "https://example.com/clean",
            "Returned after automatic protection handling",
        )])
        provider = ScraplingSearchProvider(stealth=False)
        with (
            patch.object(provider, "_fetch_basic", return_value=limited_page),
            patch.object(provider, "_fetch_stealth", return_value=clean_page) as mock_stealth,
        ):
            results = provider._search_sync("rate limited query")

        mock_stealth.assert_called_once()
        assert results[0].title == "Clean result"

    def test_native_stealth_session_enables_cloudflare_solver(self, monkeypatch, tmp_path):
        calls = {}

        class FakeSession:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                calls["closed"] = True

            def fetch(self, url, **kwargs):
                calls["fetch"] = (url, kwargs)
                page = MagicMock()
                page.html = "<html><body>clean</body></html>"
                return page

        monkeypatch.setattr(_fetchers, "StealthySession", FakeSession, raising=False)
        provider = ScraplingSearchProvider(stealth=True, user_data_dir=str(tmp_path / "profile"))
        sentinel = object()
        monkeypatch.setattr(
            "theriad.search.scrapling_provider._parse_html",
            lambda html, url: sentinel,
        )

        assert provider._fetch_stealth_browser("https://protected.example") is sentinel
        assert calls["init"]["solve_cloudflare"] is True
        assert calls["init"]["user_data_dir"].endswith("profile")
        assert calls["fetch"][1]["solve_cloudflare"] is True
        assert calls["fetch"][1]["timeout"] >= 60_000
        assert calls["closed"] is True

# ---------------------------------------------------------------------------
# async wrapper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_async_delegates_to_sync():
    provider = ScraplingSearchProvider(stealth=False)
    with patch.object(provider, "_search_sync", return_value=[]) as mock_sync:
        result = await provider.search("async test")
    mock_sync.assert_called_once_with("async test")
    assert result == []


# ---------------------------------------------------------------------------
# service.py integration: SCRAPLING_ENABLED wires up ScraplingSearchProvider
# ---------------------------------------------------------------------------

class TestServiceIntegration:
    def test_scrapling_selected_when_enabled(self, monkeypatch):
        monkeypatch.setenv("SCRAPLING_ENABLED", "1")
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("THERIAD_CHROME_PROFILE_MODE", raising=False)

        from theriad.search.service import SearchService
        svc = SearchService()
        assert isinstance(svc.provider, ScraplingSearchProvider)

    def test_searxng_preferred_over_scrapling(self, monkeypatch):
        monkeypatch.setenv("SCRAPLING_ENABLED", "1")
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8888")
        monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("THERIAD_CHROME_PROFILE_MODE", raising=False)

        from theriad.search.service import SearchService, SearXNGSearchProvider
        svc = SearchService()
        assert isinstance(svc.provider, SearXNGSearchProvider)

    def test_scrapling_is_not_selected_merely_because_it_is_installed(self, monkeypatch):
        monkeypatch.delenv("SCRAPLING_ENABLED", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("THERIAD_CHROME_PROFILE_MODE", raising=False)

        from theriad.search.service import DdgHtmlSearchProvider, SearchService
        svc = SearchService()
        assert isinstance(svc.provider, DdgHtmlSearchProvider)

    def test_scrapling_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("SCRAPLING_ENABLED", "0")
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("THERIAD_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("THERIAD_CHROME_PROFILE_MODE", raising=False)

        from theriad.search.service import DdgHtmlSearchProvider, SearchService
        svc = SearchService()
        assert isinstance(svc.provider, DdgHtmlSearchProvider)
