"""Scrapling-backed search provider — stealth SERP scraping.

Scrapes DuckDuckGo's HTML endpoint using Scrapling's fetcher backends.
Avoids the API-key requirement of Brave and the infrastructure requirement
of SearXNG, while handling fingerprinting and CAPTCHA challenges that block
plain httpx/requests clients.

Env vars
--------
SCRAPLING_ENABLED          Optional escape hatch. Set to "0" to disable the
                           automatic provider when its fetchers are installed.
SCRAPLING_STEALTH          Optional override to start browser-backed fetching
                           immediately; normal mode escalates automatically when
                           the lightweight fetcher is challenged.
CAPTCHA_SOLVER_API_KEY     CapSolver API key (capsolver.com). When set, any CAPTCHA
                           detected on a stealth fetch is solved automatically and
                           the post-challenge page is returned.
"""
from __future__ import annotations

import asyncio
import os
import urllib.parse
from pathlib import Path

from theriad.core.logging.logger import get_logger

from .captcha_solver import CaptchaSolver, detect_captcha, solve_with_playwright
from .service import SearchResult

logger = get_logger("search.scrapling")

_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_MAX_RESULTS = 8
_CLOUDFLARE_TIMEOUT_MS = 60_000
# Scrapling's documented Cloudflare timeout is 60s, but its interactive
# Turnstile retry path can recurse beyond that on a challenge it cannot clear.
# Keep the user-facing fallback bounded; a real human handoff should not wait
# forever for an invisible browser worker.
_STEALTH_WORKER_TIMEOUT_S = 30.0
_STEALTH_STATUS_CODES = frozenset({202, 403, 429, 503, 520, 521, 522, 523, 524})


def default_scrapling_profile_dir() -> str:
    """Return Theriad's private, non-user Chrome profile for web sessions."""
    configured = os.environ.get("THERIAD_SCRAPLING_PROFILE_DIR", "").strip()
    return configured or str(Path.home() / ".theriad" / "scrapling-profile")


def _decode_ddg_href(href: str) -> str:
    """Unwrap a DuckDuckGo redirect href to the real destination URL.

    DDG HTML results carry hrefs like:
        //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...
    The actual URL is the URL-decoded value of the 'uddg' query parameter.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" not in href:
        return href if href.startswith("http") else ""
    parsed = urllib.parse.urlparse(href)
    uddg = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
    return urllib.parse.unquote(uddg) if uddg else ""


def _parse_html(html: str, url: str):
    """Parse raw HTML with Scrapling's Adaptor so we can use .css() selectors."""
    from scrapling.parser import Adaptor
    return Adaptor(html, url=url, auto_match=False)


def _raw_page_html(page) -> str:
    """Extract raw HTML from Scrapling 0.3/0.4 page objects."""
    # Older fetchers exposed ``.html`` while current Response objects expose
    # the document bytes as ``.body``. Prefer both before falling back to
    # stringification; ``str(Response)`` is only a status wrapper.
    try:
        html = page.html
    except AttributeError:
        html = None
    if html:
        return html.decode(errors="replace") if isinstance(html, bytes) else str(html)
    try:
        body = page.body
    except AttributeError:
        body = None
    if body:
        return body.decode(errors="replace") if isinstance(body, bytes) else str(body)
    return str(page) or ""


def _needs_stealth(page, url: str) -> bool:
    """Detect challenge, rate-limit, or bot-block responses before parsing."""
    if page is None:
        return True
    status = getattr(page, "status", None)
    if not isinstance(status, int):
        status = getattr(page, "status_code", None)
    return (
        isinstance(status, int) and status in _STEALTH_STATUS_CODES
    ) or detect_captcha(_raw_page_html(page), url) is not None


def _run_stealth_session_worker(url: str, profile_dir: str, connection) -> None:
    """Run Scrapling's native solver in a killable child process.

    Scrapling's interactive solver can retry recursively when a challenge is
    unsolvable. A process boundary is required here because cancelling a
    Python thread cannot stop the browser or safely release the shared profile.
    """
    try:
        from scrapling.fetchers import StealthySession

        with StealthySession(
            headless=True,
            solve_cloudflare=True,
            timeout=_CLOUDFLARE_TIMEOUT_MS,
            user_data_dir=profile_dir,
            max_pages=1,
        ) as session:
            page = session.fetch(
                url,
                solve_cloudflare=True,
                timeout=_CLOUDFLARE_TIMEOUT_MS,
            )
            connection.send(("ok", _raw_page_html(page)))
    except BaseException as exc:  # child must report all failures to its parent
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        connection.close()


def _basic_fetch(url: str):
    """Fetch through Scrapling's basic client across the 0.3/0.4 APIs."""
    from scrapling.fetchers import Fetcher

    get = Fetcher.get
    # Scrapling 0.4 exposes get() as a classmethod, avoiding the deprecated
    # no-op BaseFetcher constructor. Older releases exposed an instance API.
    if getattr(get, "__self__", None) is Fetcher:
        return get(url, stealthy_headers=True, follow_redirects=True)
    return Fetcher(auto_match=False).get(url, stealthy_headers=True, follow_redirects=True)


class ScraplingSearchProvider:
    """Search provider that scrapes DuckDuckGo HTML results via Scrapling.

    Fetch modes (in escalation order):
    1. basic (default): spoofed browser headers — fast, no browser.
    2. native stealth: automatically used when the basic fetch is challenged;
       the Theriad-owned session enables Scrapling's Cloudflare solver.
    3. captcha-solve (CAPTCHA_SOLVER_API_KEY set): when native stealth still
       lands on a supported CAPTCHA, CapSolver obtains a token which is
       injected via Playwright so the bot receives the real page.
    """

    name = "scrapling"

    def __init__(self, stealth: bool = False, user_data_dir: str | None = None) -> None:
        self._stealth = stealth
        self._solver = CaptchaSolver.from_environment()
        self._user_data_dir = user_data_dir or default_scrapling_profile_dir()

    @staticmethod
    def is_available() -> bool:
        try:
            import scrapling.fetchers  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def from_environment(cls) -> ScraplingSearchProvider:
        stealth = os.environ.get("SCRAPLING_STEALTH", "").strip().lower() in {"1", "true", "yes"}
        return cls(stealth=stealth, user_data_dir=default_scrapling_profile_dir())

    async def search(self, query: str) -> list[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query)

    def fetch_html(self, url: str) -> str:
        """Fetch one page cheaply, escalating automatically if challenged."""
        page = self._fetch_stealth(url) if self._stealth else self._fetch_basic(url)
        if not self._stealth and _needs_stealth(page, url):
            page = self._fetch_stealth(url)
        return self._page_html(page) if page is not None else ""

    def _search_sync(self, query: str) -> list[SearchResult]:
        url = f"{_DDG_HTML_URL}?{urllib.parse.urlencode({'q': query})}"
        logger.debug("scrapling search query=%r stealth=%s solver=%s",
                     query, self._stealth, self._solver is not None)

        page = self._fetch_stealth(url) if self._stealth else self._fetch_basic(url)
        if not self._stealth and page is not None:
            # Keep ordinary requests cheap, but automatically escalate a
            # challenge instead of requiring SCRAPLING_STEALTH=1.
            if _needs_stealth(page, url):
                logger.info("Protected/limited basic fetch; escalating to Scrapling stealth")
                page = self._fetch_stealth(url)
        elif not self._stealth and page is None:
            # A blocked/empty basic fetch is another useful signal to try the
            # browser-backed path once before returning no results.
            page = self._fetch_stealth(url)
        if page is None:
            logger.warning(
                "DuckDuckGo search returned no page for query=%r — the request may have "
                "been blocked or timed out. Theriad already tried its automatic stealth "
                "path; check your network connectivity or use the built-in fallback.",
                query,
            )
            return []

        return self._extract_results(page)

    def _extract_results(self, page) -> list[SearchResult]:
        results: list[SearchResult] = []
        for el in page.css("div.result"):
            title_el = self._first_css(el, "a.result__a")
            snippet_el = (
                self._first_css(el, "a.result__snippet")
                or self._first_css(el, "div.result__snippet")
                or self._first_css(el, "span.result__snippet")
            )
            if not title_el:
                continue
            title = (title_el.text or "").strip()
            raw_href = title_el.attrib.get("href", "")
            real_url = _decode_ddg_href(raw_href)
            if not real_url or not real_url.startswith("http"):
                continue
            snippet = (snippet_el.text if snippet_el else "").strip()
            results.append(SearchResult(title=title, url=real_url, snippet=snippet, source=self.name))
            if len(results) >= _MAX_RESULTS:
                break
        logger.debug("scrapling extracted %d results", len(results))
        return results

    @staticmethod
    def _first_css(node, selector: str):
        """Return the first CSS match across Scrapling selector versions."""
        css_first = getattr(node, "css_first", None)
        if callable(css_first):
            return css_first(selector)
        matches = node.css(selector)
        return matches[0] if matches else None

    def _fetch_basic(self, url: str):
        """httpx-backed fetch with spoofed browser headers — no browser needed."""
        return _basic_fetch(url)

    def _fetch_stealth(self, url: str):
        """Browser-backed stealth fetch with optional CAPTCHA solving fallback."""
        page = self._fetch_stealth_browser(url)
        if page is None:
            return None

        # When solver is configured, check the page for CAPTCHA challenges and
        # solve them automatically rather than returning zero results.
        if self._solver is not None:
            html = self._page_html(page)
            challenge = detect_captcha(html, url)
            if challenge is not None:
                logger.info(
                    "CAPTCHA detected (%s) at %s — attempting autonomous solve",
                    challenge.type.name, url,
                )
                solved_html = solve_with_playwright(
                    url,
                    challenge,
                    self._solver,
                    user_data_dir=self._user_data_dir,
                )
                if solved_html:
                    return _parse_html(solved_html, url)
                logger.warning(
                    "CAPTCHA at %s was detected but could not be automatically cleared — "
                    "search results will be missing. Verify your CAPTCHA_SOLVER_API_KEY "
                    "has sufficient balance at capsolver.com, or check that the site key "
                    "was correctly extracted from the challenge page.",
                    url,
                )

        return page

    def _fetch_stealth_browser(self, url: str):
        """Use Scrapling's native Cloudflare solver with a Theriad-owned profile."""
        try:
            from scrapling.fetchers import StealthySession

            profile_dir = Path(self._user_data_dir).expanduser()
            profile_dir.mkdir(parents=True, exist_ok=True)
            # Keep unit/embedding fakes in-process. The real Scrapling class
            # runs in a child so an unresolvable interactive challenge can be
            # terminated without leaving a live browser attached to the
            # shared task profile.
            if not StealthySession.__module__.startswith("scrapling."):
                with StealthySession(
                    headless=True,
                    solve_cloudflare=True,
                    timeout=_CLOUDFLARE_TIMEOUT_MS,
                    user_data_dir=str(profile_dir),
                    max_pages=1,
                ) as session:
                    page = session.fetch(
                        url,
                        solve_cloudflare=True,
                        timeout=_CLOUDFLARE_TIMEOUT_MS,
                    )
                    html = self._page_html(page)
                return _parse_html(html, url) if html else None

            import multiprocessing

            context = multiprocessing.get_context("spawn")
            parent_connection, child_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_run_stealth_session_worker,
                args=(url, str(profile_dir), child_connection),
                daemon=True,
            )
            try:
                process.start()
                child_connection.close()
                process.join(_STEALTH_WORKER_TIMEOUT_S)
                if process.is_alive():
                    logger.warning(
                        "Scrapling stealth handling exceeded %.0fs for %s; "
                        "falling back to the task browser handoff",
                        _STEALTH_WORKER_TIMEOUT_S,
                        url,
                    )
                    process.terminate()
                    process.join(5)
                    return None
                if parent_connection.poll(0.5):
                    outcome, payload = parent_connection.recv()
                    if outcome == "ok" and payload:
                        return _parse_html(payload, url)
                    logger.debug("Scrapling native solver failed for %s: %s", url, payload)
                return None
            finally:
                parent_connection.close()
        except Exception as exc:
            logger.debug("Scrapling native Cloudflare session unavailable (%s)", exc)

        # Compatibility fallback for older Scrapling installations that do not
        # expose StealthySession or its native solver.
        try:
            from scrapling.fetchers import StealthyFetcher
            return StealthyFetcher(auto_match=False).fetch(
                url,
                headless=True,
                solve_cloudflare=True,
                timeout=_CLOUDFLARE_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.debug("StealthyFetcher unavailable (%s), trying PlayWrightFetcher", exc)
        try:
            from scrapling.fetchers import PlayWrightFetcher
            return PlayWrightFetcher(auto_match=False).fetch(
                url,
                headless=True,
                timeout=_CLOUDFLARE_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.warning(
                "Stealth fetch failed for %s — neither camoufox (StealthyFetcher) nor "
                "Playwright (PlayWrightFetcher) could load the page. Ensure at least one "
                "is available: install camoufox with 'pip install camoufox' or run "
                "'playwright install chromium'. Last error: %s",
                url, exc,
            )
        return None

    @staticmethod
    def _page_html(page) -> str:
        return _raw_page_html(page)


def fetch_stealth_html(url: str, *, user_data_dir: str | None = None) -> str | None:
    """Fetch a protected page through Scrapling for the browser-agent fallback.

    This helper returns only a bounded HTML snapshot. Callers that need to
    continue interacting with the page must reopen their live browser using
    the same Theriad-owned profile so the resulting cookies are authoritative.
    """
    provider = ScraplingSearchProvider(stealth=True, user_data_dir=user_data_dir)
    page = provider._fetch_stealth(url)
    if page is None:
        return None
    return provider._page_html(page)
