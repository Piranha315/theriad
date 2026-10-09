"""Search abstraction used before opening result URLs in Chromium.

The normal path is intentionally API/keyless-first: a local SearXNG or Brave
provider is preferred when configured, then an opt-in Scrapling provider and a
small DuckDuckGo HTML client provide bounded fallbacks. These providers return
real result pages only; they never fabricate search evidence. The browser
launch provider is retained only as a deprecated compatibility fallback.
"""
from __future__ import annotations

import html as html_lib
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .navigation import is_unresolved_search_url, resolve_initial_navigation


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str) -> list[SearchResult]: ...


class SearXNGSearchProvider:
    """Self-hosted SearXNG instance — zero per-query cost."""

    name = "searxng"

    def __init__(self, base_url: str) -> None:
        # Strip trailing slash for consistent URL building.
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "categories": "general"},
            )
            response.raise_for_status()
        rows = response.json().get("results", [])
        return [
            SearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("content") or ""),
                source=self.name,
            )
            for row in rows[:8]
            if row.get("url")
        ]


class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 8},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
            response.raise_for_status()
        rows = response.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("description") or ""),
                source=self.name,
            )
            for row in rows
            if row.get("url")
        ]


class _DdgHtmlResultParser(HTMLParser):
    """Extract the small result surface from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict[str, str]] = []
        self._field: str | None = None
        self._record: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._record = {
                "title": "",
                "url": html_lib.unescape(attrs_dict.get("href") or ""),
                "snippet": "",
            }
            self.records.append(self._record)
            self._field = "title"
        elif "result__snippet" in classes and self.records:
            self._record = self.records[-1]
            self._field = "snippet"

    def handle_data(self, data: str) -> None:
        if self._record is not None and self._field:
            self._record[self._field] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None
            self._record = None


def _decode_ddg_result_url(href: str) -> str:
    """Unwrap DuckDuckGo redirect URLs and reject non-web targets."""
    raw = html_lib.unescape((href or "").strip())
    if raw.startswith("//"):
        raw = "https:" + raw
    if "duckduckgo.com/l/" in raw:
        query = parse_qs(urlparse(raw).query)
        raw = unquote((query.get("uddg") or [""])[0])
    return raw if raw.startswith(("http://", "https://")) else ""


class DdgHtmlSearchProvider:
    """Keyless public web search used by the automatic fallback path."""

    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    async def search(self, query: str) -> list[SearchResult]:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Theriad/1.0)"},
        ) as client:
            response = await client.get(self.endpoint, params={"q": query})
            response.raise_for_status()

        parser = _DdgHtmlResultParser()
        parser.feed(response.text)
        results: list[SearchResult] = []
        for row in parser.records:
            url = _decode_ddg_result_url(row["url"])
            title = " ".join(row["title"].split())
            snippet = " ".join(row["snippet"].split())
            if not url or not title:
                continue
            results.append(SearchResult(title, url, snippet, self.name))
            if len(results) >= 8:
                break
        return results


# Title prefix of BrowserSearchProvider's single synthetic entry. It marks a
# search *launch page*, not an extracted finding — user-facing surfaces should
# detect it via is_placeholder_result() and avoid presenting it as a result.
PLACEHOLDER_TITLE_PREFIX = "Search result for "


def is_placeholder_result(result: dict) -> bool:
    """True when a search_results dict is BrowserSearchProvider's placeholder."""
    return str(result.get("title") or "").startswith(PLACEHOLDER_TITLE_PREFIX)


class BrowserSearchProvider:
    """Deprecated compatibility provider for launching a search page.

    Read-only public-web retrieval is owned by ``SearchService``'s API/keyless
    providers. This provider remains available only for explicit legacy
    fallback configuration and should not be used for new callers.
    """

    name = "browser_search"

    async def search(self, query: str) -> list[SearchResult]:
        warnings.warn(
            "BrowserSearchProvider is deprecated for read-only search; "
            "use SearchService's API/keyless providers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        navigation = resolve_initial_navigation(query)
        return [
            SearchResult(
                title=f"{PLACEHOLDER_TITLE_PREFIX}{query}",
                url=navigation.resolved_url,
                snippet=(
                    f"Browser search via {navigation.provider}. Configure Brave "
                    "Search credentials for API-native result extraction."
                ),
                source=navigation.provider,
            )
        ]


# Prompts that ask the agent to operate a page rather than do a read-only
# public-web lookup. These stay on the full-browser path even when the intent
# classifier routes them as browser_agent.
_INTERACTIVE_BROWSER_RE = re.compile(
    r"\b(go\s+to|open(?: the)? (?:website|site|page)|navigate\s+to|"
    r"brows(?:e|ing)(?: (?:the|this|that))?|visit|"
    r"summari[sz](?:e|ing)(?: this| the)? (?:website|site|page)|"
    r"click|scroll|interact|fill(?: out)?|submit|"
    r"log\s*in|sign\s*in|login|checkout|purchase|buy(?: now)?|"
    r"add to (?:cart|basket)|book|reserve|download|upload|"
    r"send (?:an )?(?:email|message)|delete|"
    r"what(?:'s| is) on (?:the )?screen|screenshot)\b",
    re.IGNORECASE,
)


def is_web_search_request(prompt: str) -> bool:
    """Kept for backward compatibility — always returns True."""
    return True


def is_api_search_request(prompt: str) -> bool:
    """Return whether a prompt is safe for read-only API-based web search.

    Any browser-classified prompt uses API search unless it contains explicit
    interactive signals (click, login, go to, etc.) that require a real browser.
    The positive keyword gate has been removed — the intent classifier decides
    whether web access is needed at all.
    """
    return not bool(_INTERACTIVE_BROWSER_RE.search(prompt or ""))


class SearchService:
    """Choose the configured search provider.

    API search is authoritative when SearXNG or Brave is configured.  A
    keyless HTML search is the automatic fallback; opening a search-engine page
    in a browser is a deprecated, explicit compatibility fallback because it
    can trigger CAPTCHA or approval flows.
    """

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self._explicit_provider = provider is not None
        self.provider = provider or self._from_environment()

    @staticmethod
    def _browser_fallback_enabled() -> bool:
        enabled = os.environ.get("THERIAD_SEARCH_FALLBACK_TO_BROWSER", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if enabled:
            warnings.warn(
                "THERIAD_SEARCH_FALLBACK_TO_BROWSER is deprecated; "
                "ordinary read-only search now fails closed instead of opening a browser.",
                DeprecationWarning,
                stacklevel=2,
            )
        return enabled

    @staticmethod
    def _from_environment() -> SearchProvider:
        # Browser profile and navigation-engine settings must not disable
        # API-first search. Keeping those settings coupled made an otherwise
        # valid browser preference silently replace structured results with a
        # launch-page placeholder. Browser navigation remains an explicit,
        # deprecated last-resort compatibility fallback in search(), never the
        # primary provider.
        # Prefer SearXNG (free, self-hosted) over everything else.
        searxng_url = os.environ.get("SEARXNG_URL", "").strip()
        if searxng_url:
            return SearXNGSearchProvider(searxng_url)
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
        if brave_key:
            return BraveSearchProvider(brave_key)
        # Scrapling can launch a stealth browser and wait on challenge pages;
        # installation alone must not put that heavyweight path in every
        # request. Keep it explicit and bounded.
        scrapling_setting = os.environ.get("SCRAPLING_ENABLED", "").strip().lower()
        if scrapling_setting in {"1", "true", "yes", "on"}:
            from .scrapling_provider import ScraplingSearchProvider
            if ScraplingSearchProvider.is_available():
                return ScraplingSearchProvider.from_environment()
        return DdgHtmlSearchProvider()

    async def search(self, query: str) -> list[SearchResult]:
        """Return usable result pages, recovering from stale search endpoints.

        A configured metasearch endpoint can reply successfully with a search
        homepage (or an unresolved Google/Bing URL) instead of result pages.
        Treat that as a provider failure and use the same keyless fallback as a
        transport error; otherwise the caller receives a navigation target and
        the user-facing search lane appears to have succeeded without evidence.
        """
        try:
            provider_results = await self._provider_search(self.provider, query)
            results = self._usable_results(provider_results)
            if not results:
                # Explicit provider injection is also the compatibility seam
                # used by the browser-navigation path. Preserve its historic
                # rewrite behavior for a provider that returns only a search
                # homepage, while configured automatic providers fail over to
                # real keyless result pages below.
                if self._explicit_provider and provider_results:
                    return await BrowserSearchProvider().search(query)
                raise RuntimeError("search provider returned no usable result pages")
        except Exception as primary_error:
            # A configured provider may be stale or offline.  Keep the user
            # out of provider configuration by trying the built-in path once.
            if not self._explicit_provider and not isinstance(self.provider, DdgHtmlSearchProvider):
                try:
                    results = self._usable_results(await self._keyless_search(query))
                except Exception:
                    results = []
            else:
                results = []

            if results:
                return results
            if self._browser_fallback_enabled():
                return await BrowserSearchProvider().search(query)
            raise RuntimeError(
                "Automatic web search is unavailable right now. "
                "Check the network connection and try again."
            ) from primary_error

        if results:
            return results
        if not self._explicit_provider and not self._browser_fallback_enabled():
            raise RuntimeError(
                "Automatic web search returned no usable results. Try a broader question."
            )
        return await BrowserSearchProvider().search(query)

    @staticmethod
    def _usable_results(results: list[SearchResult]) -> list[SearchResult]:
        """Discard blank and unresolved search-navigation targets."""
        return [
            result for result in results
            if str(getattr(result, "url", "") or "").startswith(("http://", "https://"))
            and not is_unresolved_search_url(result.url)
        ]

    async def fetch(self, url: str) -> dict:
        """Fetch bounded public-page evidence through the canonical service."""
        from .evidence import fetch_page_evidence

        return await fetch_page_evidence(url)

    @classmethod
    async def _keyless_search(cls, query: str) -> list[SearchResult]:
        """Try the keyless endpoint at most twice for transient empty pages."""
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                return await cls._provider_search(DdgHtmlSearchProvider(), query)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    @staticmethod
    async def _provider_search(
        provider: SearchProvider, query: str,
    ) -> list[SearchResult]:
        """Call a network provider through the shared bounded circuit."""
        from theriad.runtime.circuit_breaker import get_circuit_registry

        key = f"search:{type(provider).__name__}"
        registry = get_circuit_registry()
        registry.before_call(key)
        started = time.perf_counter()
        try:
            results = await provider.search(query)
            if not results:
                raise RuntimeError("search provider returned no results")
        except Exception as exc:
            registry.record_failure(
                key,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        registry.record_success(
            key,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return results
