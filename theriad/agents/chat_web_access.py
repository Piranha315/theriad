"""Gates for model-driven internet access in chat.

Decides, per request, whether the chat model gets web tools and whether an
OpenSandbox browser may be offered. These are capability and safety gates, not
routing rules: the learned router still chooses the model.
"""
from __future__ import annotations

import os
import time
from typing import Any

from theriad.agents.web_tool_loop import SandboxBrowserTools

_AVAILABILITY_TTL_SECONDS = 30.0
_availability_cache: tuple[float, bool] | None = None


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def web_tools_enabled(*, guest: bool, execution_provider: str) -> bool:
    """Web tools need the user's own OpenRouter key: search is billed to it."""
    if guest or not _flag("THERIAD_CHAT_WEB_TOOLS", True):
        return False
    return execution_provider in {"theriad", "openrouter"}


def model_supports_tools(model_slug: str) -> bool:
    """Use the live OpenRouter catalog; an unknown catalog allows an attempt."""
    try:
        from theriad.accounting.live_prices import get_live_snapshot

        known = get_live_snapshot().tool_models
    except Exception:
        return True
    if not known:
        return True
    return model_slug.split(":", 1)[0] in known


def _max_sessions() -> int:
    try:
        return max(0, int(os.environ.get("THERIAD_OPENSANDBOX_MAX_SESSIONS", "4")))
    except ValueError:
        return 4


def browser_allowed_here() -> bool:
    from theriad.api.auth import is_hosted_deployment

    if is_hosted_deployment():
        return _flag("THERIAD_HOSTED_WEB_BROWSING", False)
    return _flag("THERIAD_CHAT_BROWSER_TOOLS", True)


async def _sandbox_healthy(runtime: Any) -> bool:
    global _availability_cache
    now = time.monotonic()
    if _availability_cache and now - _availability_cache[0] < _AVAILABILITY_TTL_SECONDS:
        return _availability_cache[1]
    try:
        healthy = bool((await runtime.availability(probe=True)).get("available"))
    except Exception:
        healthy = False
    _availability_cache = (now, healthy)
    return healthy


def reset_for_tests() -> None:
    global _availability_cache
    _availability_cache = None


async def browser_tools_for_turn(
    *,
    goal: str,
    owner_user_id: str | None,
    owner_organization_id: str | None,
    runtime: Any = None,
) -> SandboxBrowserTools | None:
    """Offer one sandbox browser for this turn when allowed, healthy, and under caps."""
    if not browser_allowed_here():
        return None
    if runtime is None:
        try:
            from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

            runtime = get_opensandbox_browser_runtime()
        except Exception:
            return None
    if not await _sandbox_healthy(runtime):
        return None
    sessions = list(getattr(runtime, "sessions", {}).values())
    live = [s for s in sessions if getattr(s, "status", "") not in {"stopped", "completed", "failed"}]
    if len(live) >= _max_sessions():
        return None
    if owner_user_id and any(getattr(s, "owner_user_id", None) == owner_user_id for s in live):
        return None
    return SandboxBrowserTools(
        runtime,
        goal=goal,
        owner_user_id=owner_user_id,
        owner_organization_id=owner_organization_id,
    )


def browsing_status(*, probe: bool = True) -> dict[str, Any]:
    """Secret-free readiness of chat browsing, for preflight and capabilities."""
    enabled = browser_allowed_here()
    status: dict[str, Any] = {
        "enabled": enabled,
        "max_sessions": _max_sessions(),
        "sandbox_healthy": None,
    }
    if enabled and probe:
        import httpx

        domain = os.environ.get("THERIAD_OPENSANDBOX_DOMAIN") or os.environ.get("OPENSANDBOX_DOMAIN") or "127.0.0.1:8080"
        protocol = os.environ.get("THERIAD_OPENSANDBOX_PROTOCOL") or "http"
        try:
            response = httpx.get(f"{protocol}://{domain}/health", timeout=3.0)
            status["sandbox_healthy"] = response.status_code == 200 and "healthy" in response.text
        except Exception:
            status["sandbox_healthy"] = False
    return status
