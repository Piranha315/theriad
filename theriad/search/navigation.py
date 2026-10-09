"""Deterministic natural-language prompt to initial browser navigation."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote_plus, urlparse


@dataclass(frozen=True)
class InitialNavigation:
    original_prompt: str
    resolved_url: str
    provider: str
    reason: str
    action: str = "navigate"

    def to_dict(self) -> dict:
        return {
            "original_prompt": self.original_prompt,
            "resolved_url": self.resolved_url,
            "provider": self.provider,
            "reason": self.reason,
            "action": self.action,
        }


_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)*\.[a-z]{2,})\b",
    re.IGNORECASE,
)
_KNOWN_SITES = {
    "wikipedia": "https://www.wikipedia.org/",
    "zillow": "https://www.zillow.com/",
    "redfin": "https://www.redfin.com/",
    "realtor": "https://www.realtor.com/",
    "apartments": "https://www.apartments.com/",
    "craigslist": "https://www.craigslist.org/",
    "github": "https://github.com/",
    "linkedin": "https://www.linkedin.com/",
    "youtube": "https://www.youtube.com/",
    "amazon": "https://www.amazon.com/",
}
_VALID_PROVIDERS = frozenset({"google", "bing", "duckduckgo"})


def configured_search_provider() -> str:
    provider = os.environ.get("THERIAD_SEARCH_PROVIDER", "").strip().lower()
    return provider if provider in _VALID_PROVIDERS else "google"


def build_search_url(query: str, provider: str | None = None) -> str:
    selected = provider or configured_search_provider()
    cleaned_query = (query or "").strip() or "Search the web"
    encoded = quote_plus(cleaned_query)
    if selected == "bing":
        return f"https://www.bing.com/search?q={encoded}"
    if selected == "duckduckgo":
        return f"https://html.duckduckgo.com/html/?q={encoded}"
    return f"https://www.google.com/search?q={encoded}"


def is_placeholder_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host == "example.com" or host.endswith(".example.com")


def is_unresolved_search_url(url: str | None) -> bool:
    """True for generic search homepages or search URLs with no usable query."""
    if not url:
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return False

    is_search_host = host in {
        "google.com", "www.google.com", "bing.com", "www.bing.com",
    }
    if not is_search_host:
        return False

    params = parse_qs(parsed.query)
    query = (params.get("q") or [""])[0].strip()
    return not query


def resolve_initial_navigation(
    user_prompt: str, *, provider: str | None = None,
) -> InitialNavigation:
    """Resolve a prompt to a real site or search URL; never return example.com."""
    prompt = (user_prompt or "").strip()
    explicit = _URL_RE.search(prompt)
    if explicit and not is_placeholder_url(explicit.group(0)):
        url = explicit.group(0).rstrip(".,;:!?)")
        return InitialNavigation(prompt, url, "direct", "Prompt contains an explicit URL.")

    domain = _DOMAIN_RE.search(prompt)
    if domain:
        url = f"https://{domain.group(1).lower()}"
        if not is_placeholder_url(url):
            return InitialNavigation(prompt, url, "direct", "Prompt contains a domain.")

    lowered = prompt.lower()
    for name, url in _KNOWN_SITES.items():
        if name in lowered:
            return InitialNavigation(prompt, url, name, f"Prompt names {name}.")

    selected_provider = provider or configured_search_provider()
    return InitialNavigation(
        original_prompt=prompt,
        resolved_url=build_search_url(prompt, selected_provider),
        provider=selected_provider,
        reason=f"Resolved natural-language web task through {selected_provider} search.",
    )


# Product-spec spelling for callers that mirror the frontend naming.
resolveInitialNavigation = resolve_initial_navigation
