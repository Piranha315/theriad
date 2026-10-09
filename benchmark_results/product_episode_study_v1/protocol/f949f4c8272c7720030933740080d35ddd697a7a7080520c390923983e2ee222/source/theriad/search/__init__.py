"""API-first web search providers."""

from .service import (
    DdgHtmlSearchProvider,
    SearchResult,
    SearchService,
    SearXNGSearchProvider,
    is_api_search_request,
    is_web_search_request,
)
from .scrapling_provider import (
    ScraplingSearchProvider,
    default_scrapling_profile_dir,
    fetch_stealth_html,
)
from .captcha_solver import CaptchaSolver, CaptchaChallenge, CaptchaType, detect_captcha
from .navigation import (
    InitialNavigation,
    build_search_url,
    is_placeholder_url,
    is_unresolved_search_url,
    resolveInitialNavigation,
    resolve_initial_navigation,
)

__all__ = [
    "CaptchaChallenge",
    "CaptchaSolver",
    "CaptchaType",
    "DdgHtmlSearchProvider",
    "InitialNavigation",
    "ScraplingSearchProvider",
    "default_scrapling_profile_dir",
    "SearchResult",
    "SearchService",
    "SearXNGSearchProvider",
    "build_search_url",
    "detect_captcha",
    "fetch_stealth_html",
    "is_api_search_request",
    "is_placeholder_url",
    "is_unresolved_search_url",
    "is_web_search_request",
    "resolveInitialNavigation",
    "resolve_initial_navigation",
]
