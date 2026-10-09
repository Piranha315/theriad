"""Public-page evidence acquisition for web research.

Search-result snippets are discovery hints, not proof. This module fetches a
bounded, public page snapshot for the best results and extracts visible text so
the answer writer can distinguish an actual source from an empty or blocked
search response. It never reads a user's browser profile or cookies.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from .captcha_solver import detect_captcha

MAX_PAGE_TEXT_CHARS = 5_000
MAX_EVIDENCE_PAGES = 6
_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "head"})
_BLOCKED_STATUS_CODES = frozenset({401, 403, 429, 451, 500, 502, 503, 504})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5


class _BlockedPublicUrlError(ValueError):
    """Raised when a URL or one of its resolved redirect targets is unsafe."""


def is_public_web_url(url: str) -> bool:
    """Reject obvious local/private targets before server-side acquisition.

    DNS-backed hosts are resolved immediately before acquisition by
    ``_assert_resolved_public_url``. Keeping this synchronous helper limited to
    URL syntax and literal addresses also makes it safe for callers that only
    need to classify a candidate without performing network I/O.
    """
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        try:
            port = parsed.port
        except ValueError:
            return False
        if port == 0:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True
        return _is_public_ip(address)
    except (ValueError, UnicodeError):
        return False


def _is_public_ip(address: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether an address is globally routable, not merely non-private."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return bool(
        parsed.is_global
        and not parsed.is_private
        and not parsed.is_loopback
        and not parsed.is_link_local
        and not parsed.is_multicast
        and not parsed.is_reserved
        and not parsed.is_unspecified
    )


async def _assert_resolved_public_url(url: str) -> None:
    """Validate syntax and every DNS answer before opening an HTTP connection."""
    if not is_public_web_url(url):
        raise _BlockedPublicUrlError("URL is not an allowed public HTTP(S) target")

    parsed = urlsplit(url)
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            answers = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise _BlockedPublicUrlError("Could not resolve a public HTTP(S) target") from exc

        addresses = {
            str(item[4][0]).split("%", 1)[0]
            for item in answers
            if len(item) >= 5 and item[4]
        }
        if not addresses or any(not _is_public_ip(address) for address in addresses):
            raise _BlockedPublicUrlError(
                "Resolved HTTP(S) target is not public"
            ) from None
        return

    if not _is_public_ip(literal):
        raise _BlockedPublicUrlError("Resolved HTTP(S) target is not public")


class _VisibleTextParser(HTMLParser):
    """Extract readable text without depending on BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in {"br", "p", "li", "div", "article", "section", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in _SKIP_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def extract_visible_text(html: str, *, max_chars: int = MAX_PAGE_TEXT_CHARS) -> str:
    """Return normalized visible text from a bounded HTML document."""
    parser = _VisibleTextParser()
    try:
        parser.feed(html or "")
    except Exception:
        # Malformed HTML is common; retain whatever the parser collected.
        pass
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:max_chars]


async def _fetch_http(url: str) -> tuple[int | None, str]:
    async with httpx.AsyncClient(
        timeout=10.0,
        follow_redirects=False,
        # Do not let a process-level proxy bypass the URL/IP validation above.
        trust_env=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Theriad/1.0)"},
    ) as client:
        current_url = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            await _assert_resolved_public_url(current_url)
            response = await client.get(current_url)
            if response.status_code not in _REDIRECT_STATUS_CODES:
                return response.status_code, response.text

            location = str(response.headers.get("location") or "").strip()
            if not location:
                return response.status_code, response.text
            if redirect_count >= _MAX_REDIRECTS:
                raise _BlockedPublicUrlError("Too many redirects while fetching public evidence")
            current_url = urljoin(current_url, location)
        raise _BlockedPublicUrlError("Too many redirects while fetching public evidence")


async def fetch_page_evidence(url: str) -> dict[str, Any]:
    """Fetch one public result and return a safe evidence record."""
    started = asyncio.get_running_loop().time()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "url": url,
        "retrieved_at": retrieved_at,
        "status": "unavailable",
        "status_code": None,
        "text": "",
    }
    if not is_public_web_url(url):
        result["status"] = "blocked_url"
        return result

    html = ""
    try:
        status_code, html = await _fetch_http(url)
        result["status_code"] = status_code
        if status_code in _BLOCKED_STATUS_CODES or detect_captcha(html, url):
            result["status"] = "blocked"
        elif status_code is not None and status_code < 400:
            result["status"] = "fetched"
        else:
            result["status"] = "http_error"
    except _BlockedPublicUrlError:
        # Do not invoke the optional browser-backed fallback after a URL or
        # redirect was rejected; it cannot provide the same per-hop guarantee.
        result["status"] = "blocked_url"
    except Exception as exc:
        result["status"] = "fetch_error"
        result["error_type"] = type(exc).__name__

    # Scrapling is an explicit acquisition fallback for a challenged public
    # page, not a CAPTCHA guarantee. Merely installing the optional dependency
    # must not launch stealth browser workers during ordinary research.
    if result["status"] in {"blocked", "fetch_error", "http_error"} or not html.strip():
        try:
            setting = os.environ.get("SCRAPLING_ENABLED", "").strip().lower()
            if setting in {"1", "true", "yes", "on"}:
                from .scrapling_provider import ScraplingSearchProvider

                if ScraplingSearchProvider.is_available():
                    provider = ScraplingSearchProvider.from_environment()
                    html = await asyncio.to_thread(provider.fetch_html, url)
                    if html:
                        result["status"] = "fetched_stealth"
        except Exception as exc:
            result["fallback_error_type"] = type(exc).__name__

    # PDF and other binary payloads are not HTML evidence. Feeding their raw
    # bytes through HTMLParser produces garbage such as ``%PDF-1.6`` in the
    # user-facing result and consumes the synthesis context without adding
    # readable facts. Keep the search snippet, but leave page text empty.
    if html.lstrip().startswith(("%PDF-", "PK\x03\x04")):
        html = ""
        result["status"] = "non_html"
    result["text"] = extract_visible_text(html)
    if result["status"] in {"fetched", "fetched_stealth"} and not result["text"]:
        result["status"] = "empty"
    result["elapsed_ms"] = round((asyncio.get_running_loop().time() - started) * 1000, 1)
    # Do not return exception text, cookies, headers, or raw HTML to the model.
    return result


async def fetch_results_evidence(
    results: list[dict[str, Any]],
    *,
    limit: int = MAX_EVIDENCE_PAGES,
) -> list[dict[str, Any]]:
    """Fetch a bounded set of result pages concurrently, preserving order."""
    candidates = [
        result for result in results
        if str(result.get("url") or "").startswith(("http://", "https://"))
    ][:limit]
    if not candidates:
        return []
    fetched = await asyncio.gather(
        *(fetch_page_evidence(str(result["url"])) for result in candidates),
        return_exceptions=True,
    )
    evidence: list[dict[str, Any]] = []
    for result, item in zip(candidates, fetched):
        if isinstance(item, Exception):
            evidence.append({
                "url": result.get("url"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "status": "fetch_error",
                "status_code": None,
                "text": "",
                "error_type": type(item).__name__,
            })
        else:
            evidence.append(item)
    return evidence
