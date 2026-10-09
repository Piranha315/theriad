"""OpenSandbox-backed browser runtime.

OpenSandbox is the only browser execution backend exposed by the current
product path.  It provisions the official OpenSandbox Chrome image and
connects to its DevTools endpoint from the Theriad process.  Theriad keeps the
policy and approval boundary here; page content is observation data only.

The OpenSandbox SDK and Playwright are optional imports so the API can start
and report a truthful ``unavailable`` state when the sandbox service is not
configured.  There is deliberately no synthetic-browser fallback.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from theriad.search import (
    is_placeholder_url,
    is_unresolved_search_url,
    resolve_initial_navigation,
)

from .policy import BrowserPolicy


OPEN_SANDBOX_BACKEND = "opensandbox"
DEFAULT_BROWSER_IMAGE = "opensandbox/chrome:latest"
DEFAULT_BROWSER_PORT = 9222
DEFAULT_SANDBOX_DOMAIN = "localhost:8080"
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 300
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
MAX_VISIBLE_TEXT = 4_000
MAX_ELEMENTS = 60
MAX_SCREENSHOT_BYTES = 4_000_000
MAX_TERMINAL_SNAPSHOTS = 256
LIVE_SESSION_STATUSES = frozenset({"running", "waiting_for_approval"})
TERMINAL_SESSION_STATUSES = frozenset({"completed", "failed", "stopped"})

_SENSITIVE_ACTIONS = {
    "type",
    "fill",
    "submit",
    "submit_form",
    "login",
    "send_message",
    "send_email",
    "purchase",
    "payment",
    "delete",
    "edit_data",
    "upload",
    "download",
    "install",
    "access_sensitive_account",
    "reveal_secret",
    "copy_secret",
}

_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore your instructions",
    "reveal api key",
    "reveal secret",
    "expose credentials",
    "bypass approval",
    "override policy",
    "send credentials",
    "leak secrets",
    "print api key",
    "print openrouter",
    "disregard your goal",
    "new goal:",
    "change goal to",
    "forget your",
    "act as if",
)

_SENSITIVE_KEY_MARKERS = (
    "password",
    "passcode",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "private_key",
)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _sandbox_domain() -> str:
    return _env_first("THERIAD_OPENSANDBOX_DOMAIN", "OPEN_SANDBOX_DOMAIN") or DEFAULT_SANDBOX_DOMAIN


def _sandbox_protocol() -> str:
    return (_env_first("THERIAD_OPENSANDBOX_PROTOCOL") or "http").lower()


def _sandbox_api_key() -> str | None:
    # The key is server-only.  It is never returned in status payloads or
    # copied into sandbox metadata.
    return _env_first("THERIAD_OPENSANDBOX_API_KEY", "OPEN_SANDBOX_API_KEY")


def _sandbox_image() -> str:
    return _env_first("THERIAD_OPENSANDBOX_BROWSER_IMAGE") or DEFAULT_BROWSER_IMAGE


def _sandbox_timeout() -> int:
    raw = _env_first("THERIAD_OPENSANDBOX_TIMEOUT_SECONDS")
    try:
        value = int(raw) if raw else DEFAULT_SANDBOX_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_SANDBOX_TIMEOUT_SECONDS
    return max(30, min(value, 86_400))


def _request_timeout() -> float:
    raw = _env_first("THERIAD_OPENSANDBOX_REQUEST_TIMEOUT_SECONDS")
    try:
        value = float(raw) if raw else DEFAULT_REQUEST_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_REQUEST_TIMEOUT_SECONDS
    return max(1.0, min(value, 300.0))


def _health_url() -> str:
    domain = _sandbox_domain().rstrip("/")
    if not domain.startswith(("http://", "https://")):
        domain = f"{_sandbox_protocol()}://{domain}"
    return f"{domain}/health"


def _redact_error(error: BaseException) -> str:
    message = str(error).strip() or error.__class__.__name__
    for secret in (_sandbox_api_key(),):
        if secret:
            message = message.replace(secret, "<redacted>")
    message = re.sub(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    return message[:600]


def _redact_text(value: Any, *, private_values: tuple[str, ...] = ()) -> str:
    """Return display-safe text without retaining common credential material."""
    text = str(value or "")
    for private_value in private_values:
        if len(private_value) >= 3:
            text = text.replace(private_value, "<redacted-input>")
    text = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:sk|pk|ghp|github_pat|sb|bb_live)[-_][A-Za-z0-9._-]{8,}\b",
        "<redacted-key>",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:code|key|token|secret|password|passwd|auth|authorization)=)[^&#\s]+",
        r"\1<redacted>",
        text,
    )
    return text


def _normalize_url(url: str) -> tuple[str | None, str | None]:
    candidate = (url or "").strip()
    if candidate == "about:blank" or candidate.startswith("data:text/html,"):
        return candidate, None
    parsed = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
    if parsed.scheme.lower() not in {"http", "https"}:
        return None, f"Navigation blocked: unsupported URL scheme '{parsed.scheme}:'."
    if not parsed.netloc:
        return None, "Navigation blocked: HTTP(S) URL must include a host."
    return candidate if "://" in candidate else f"https://{candidate}", None


def _cdp_url(endpoint: Any) -> str:
    value = str(getattr(endpoint, "endpoint", endpoint)).rstrip("/")
    if value.endswith("/json"):
        value = value[:-5].rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"{_sandbox_protocol()}://{value}"
    return value


def _sandbox_id(sandbox: Any) -> str | None:
    """Read the SDK sandbox identifier without depending on one SDK model."""
    if sandbox is None:
        return None
    value = getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)
    return str(value) if value is not None else None


def _endpoint_headers(endpoint: Any) -> dict[str, str]:
    raw = getattr(endpoint, "headers", None)
    if not raw:
        return {}
    # OpenSandbox 0.1.x exposes endpoint headers as an attrs model with
    # ``to_dict``/``additional_properties``; test doubles and older SDKs may
    # expose a normal mapping instead. Keep this conversion in-memory only:
    # these headers can contain the route token needed for CDP access.
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    elif hasattr(raw, "additional_properties"):
        raw = raw.additional_properties
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _contains_prompt_injection(text: str) -> bool:
    lower = (text or "").lower()
    return bool(lower) and any(phrase in lower for phrase in _INJECTION_PHRASES)


def _action_name(action: dict[str, Any] | None) -> str:
    raw = (action or {}).get("action") or "observe"
    return str(raw).strip().lower().replace(" ", "_")


def _safe_action_payload(
    action: dict[str, Any], *, private_values: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Copy action metadata without retaining nested typed values."""

    def clean(value: Any, key: str = "") -> Any:
        normalized_key = key.lower().replace("-", "_")
        if any(marker in normalized_key for marker in _SENSITIVE_KEY_MARKERS):
            return {"redacted": True, "length": len(str(value))} if value is not None else None
        if normalized_key in {"text", "value", "input", "form_data", "fields", "body"}:
            return {"redacted": True, "length": len(str(value))} if value is not None else None
        if isinstance(value, dict):
            return {
                str(child_key): clean(child_value, str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value[:50]]
        if isinstance(value, str):
            return _redact_text(value, private_values=private_values)
        return value

    return {
        str(key): clean(value, str(key))
        for key, value in action.items()
        if key not in {"session_id", "approved", "deny"}
    }


def _normalize_allowed_domains(domains: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a task domain allowlist without accepting URL paths or schemes."""
    normalized: set[str] = set()
    for raw in domains or []:
        candidate = str(raw or "").strip().lower()
        if not candidate:
            continue
        if "://" in candidate:
            candidate = urlsplit(candidate).hostname or ""
        candidate = candidate.split("/", 1)[0].split(":", 1)[0].lstrip("*.")
        if candidate and re.fullmatch(r"[a-z0-9._-]+", candidate):
            normalized.add(candidate.rstrip("."))
    return sorted(normalized)


def _host_allowed(url: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return True
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _safe_observation_elements(
    elements: Any, *, private_values: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Keep useful element metadata while excluding any live form values."""
    safe: list[dict[str, Any]] = []
    for element in elements if isinstance(elements, list) else []:
        if not isinstance(element, dict):
            continue
        row: dict[str, Any] = {}
        for key in ("id", "tag", "type", "in_form", "download"):
            if key in element:
                row[key] = element[key]
        for key in ("text", "label", "placeholder", "href"):
            if key in element:
                row[key] = _redact_text(element[key], private_values=private_values)[:300]
        safe.append(row)
    return safe[:MAX_ELEMENTS]


@dataclass
class OpenSandboxBrowserSession:
    session_id: str
    sandbox: Any
    playwright: Any
    browser: Any
    context: Any
    page: Any
    visible: bool
    created_at: float
    sandbox_id: str | None = None
    owner_user_id: str | None = None
    owner_organization_id: str | None = None
    allowed_domains: list[str] = field(default_factory=list)
    user_prompt: str = ""
    status: str = "running"
    current_url: str | None = None
    page_title: str | None = None
    page_summary: str = ""
    screenshot_base64: str | None = None
    pending_action: dict[str, Any] | None = None
    # The public pending_action is redacted. Keep the original action only in
    # process memory so approval can execute it without returning typed values.
    pending_action_private: dict[str, Any] | None = field(default=None, repr=False)
    final_result: str | None = None
    error: str | None = None
    captcha_detected: bool = False
    captcha_reason: str | None = None
    resolved_initial_url: str | None = None
    navigation_reason: str | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    elements: list[dict[str, Any]] = field(default_factory=list)
    # Keep submitted values only in process memory so pages that reflect an
    # input cannot expose it in a later observation. This is never checkpointed.
    private_input_values: list[str] = field(default_factory=list, repr=False)


class OpenSandboxBrowserRuntime:
    """Manage one isolated OpenSandbox Chrome instance per browser session."""

    def __init__(
        self,
        *,
        sandbox_factory: Callable[..., Awaitable[Any]] | None = None,
        playwright_factory: Callable[[], Any] | None = None,
        policy: BrowserPolicy | None = None,
    ) -> None:
        self.policy = policy or BrowserPolicy()
        self.sessions: dict[str, OpenSandboxBrowserSession] = {}
        self._lock = asyncio.Lock()
        self._sandbox_factory = sandbox_factory
        self._playwright_factory = playwright_factory
        # Terminal sessions are removed from ``sessions`` after remote cleanup.
        # Keep a small, redacted, owner-scoped cache so status/stop remains
        # useful and idempotent without retaining browser handles or secrets.
        self._terminal_snapshots: dict[
            str, tuple[str | None, str | None, dict[str, Any]]
        ] = {}

    def _active_session_count(self) -> int:
        return sum(
            session.status in LIVE_SESSION_STATUSES for session in self.sessions.values()
        )

    def _cached_terminal_snapshot(
        self,
        session_id: str,
        *,
        owner_user_id: str | None,
        owner_organization_id: str | None,
    ) -> dict[str, Any] | None:
        cached = self._terminal_snapshots.get(session_id)
        if cached is None:
            return None
        cached_user, cached_org, payload = cached
        if cached_user != owner_user_id or cached_org != owner_organization_id:
            return None
        return copy.deepcopy(payload)

    def _cache_terminal_snapshot(self, session: OpenSandboxBrowserSession) -> None:
        if session.status not in TERMINAL_SESSION_STATUSES:
            return
        payload = self._snapshot(session)
        self._terminal_snapshots[session.session_id] = (
            session.owner_user_id,
            session.owner_organization_id,
            copy.deepcopy(payload),
        )
        while len(self._terminal_snapshots) > MAX_TERMINAL_SNAPSHOTS:
            oldest = next(iter(self._terminal_snapshots))
            self._terminal_snapshots.pop(oldest, None)

    def _dependency_status(self) -> dict[str, Any]:
        result = self.dependency_status()
        # Injected factories are deterministic seams for tests and for
        # application-managed runtimes. They must not require the optional
        # packages to be importable in this process.
        if self._sandbox_factory is not None:
            result["opensandbox_installed"] = True
        if self._playwright_factory is not None:
            result["playwright_installed"] = True
        if result.get("opensandbox_installed") and result.get("playwright_installed"):
            result.update({"available": True, "error": None})
        return result

    @staticmethod
    def dependency_status() -> dict[str, Any]:
        try:
            import opensandbox  # noqa: F401
        except ImportError:
            return {
                "available": False,
                "error": "OpenSandbox SDK is not installed. Install Theriad with the browser extra.",
                "opensandbox_installed": False,
                "playwright_installed": False,
            }
        try:
            import playwright.async_api  # noqa: F401
        except ImportError:
            return {
                "available": False,
                "error": "Playwright is not installed. Install Theriad with the browser extra.",
                "opensandbox_installed": True,
                "playwright_installed": False,
            }
        return {
            "available": True,
            "error": None,
            "opensandbox_installed": True,
            "playwright_installed": True,
        }

    async def availability(self, *, probe: bool = True) -> dict[str, Any]:
        result = self._dependency_status()
        result.update(
            {
                "backend": OPEN_SANDBOX_BACKEND,
                "server": _sandbox_domain(),
                "image": _sandbox_image(),
                "active_sessions": len(self.sessions),
            }
        )
        if not result["available"] or not probe:
            return result
        # A complete injected runtime is already the health boundary for the
        # caller. Probing localhost here would make unit tests and embedded
        # SDK integrations depend on an unrelated server.
        if self._sandbox_factory is not None and self._playwright_factory is not None:
            return result

        try:
            headers = {}
            if _sandbox_api_key():
                headers["OPEN-SANDBOX-API-KEY"] = _sandbox_api_key() or ""
            async with httpx.AsyncClient(timeout=min(_request_timeout(), 5.0)) as client:
                response = await client.get(_health_url(), headers=headers)
            if response.status_code >= 400:
                result.update(
                    {
                        "available": False,
                        "error": f"OpenSandbox health check returned HTTP {response.status_code}.",
                    }
                )
        except Exception as exc:
            result.update(
                {
                    "available": False,
                    "error": f"OpenSandbox server is unreachable: {_redact_error(exc)}",
                }
            )
        return result

    def _connection_config(self) -> Any:
        from opensandbox.config import ConnectionConfig

        return ConnectionConfig(
            domain=_sandbox_domain(),
            protocol=_sandbox_protocol(),
            api_key=_sandbox_api_key(),
            request_timeout=timedelta(seconds=_request_timeout()),
            # The Theriad backend may be remote from the sandbox runtime.  The
            # server proxy gives it a stable endpoint in that deployment shape.
            use_server_proxy=(
                (_env_first("THERIAD_OPENSANDBOX_USE_SERVER_PROXY") or "true").lower()
                in {"1", "true", "yes"}
            ),
        )

    async def _create_sandbox(self) -> Any:
        kwargs = {
            "image": _sandbox_image(),
            "timeout": timedelta(seconds=_sandbox_timeout()),
            "entrypoint": ["/entrypoint"],
            "metadata": {"theriad.component": "browser"},
        }
        if self._sandbox_factory is not None:
            created = self._sandbox_factory(**kwargs)
            return await created if inspect.isawaitable(created) else created
        # Keep the SDK import and connection config on the real path only.
        # Injected factories are intentionally usable in environments that do
        # not install the optional OpenSandbox extra.
        kwargs["connection_config"] = self._connection_config()
        from opensandbox import Sandbox

        return await Sandbox.create(**kwargs)

    def _playwright(self) -> Any:
        if self._playwright_factory is not None:
            return self._playwright_factory()
        from playwright.async_api import async_playwright

        return async_playwright()

    @staticmethod
    def _owned(
        session: OpenSandboxBrowserSession,
        owner_user_id: str | None,
        owner_organization_id: str | None,
    ) -> bool:
        return (
            session.owner_user_id == owner_user_id
            and session.owner_organization_id == owner_organization_id
        )

    def _session(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> OpenSandboxBrowserSession:
        session = self.sessions.get(session_id)
        if session is None or not self._owned(session, owner_user_id, owner_organization_id):
            raise KeyError("Browser session not found.")
        return session

    @staticmethod
    def _log(
        session: OpenSandboxBrowserSession,
        action: str,
        status: str,
        *,
        target: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "action": _redact_text(action),
            "status": status,
        }
        if target:
            entry["target"] = _redact_text(target)
        if session.current_url:
            entry["url"] = _redact_text(session.current_url)
        if error:
            entry["error"] = _redact_text(error)
        if details:
            entry["details"] = _safe_action_payload(
                details,
                private_values=tuple(session.private_input_values),
            )
        session.action_log.append(entry)
        del session.action_log[:-100]

    @staticmethod
    def _is_captcha(url: str | None, title: str | None, text: str) -> bool:
        haystack = " ".join((url or "", title or "", text)).lower()
        return any(
            marker in haystack
            for marker in (
                "captcha",
                "verify you are human",
                "i am not a robot",
                "cloudflare verification",
                "security check",
            )
        )

    async def start_session(
        self,
        *,
        session_id: str | None = None,
        visible: bool = False,
        start_url: str | None = None,
        viewport: dict[str, int] | None = None,
        user_prompt: str = "",
        allowed_domains: list[str] | None = None,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        existing = self.sessions.get(sid)
        if existing is not None:
            return await self.status(
                sid,
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
            )

        normalized_allowed_domains = _normalize_allowed_domains(allowed_domains)

        if _contains_prompt_injection(user_prompt):
            return {
                "ok": False,
                "status": "blocked",
                "backend": OPEN_SANDBOX_BACKEND,
                "error": "Goal contains suspicious content and was rejected.",
            }

        initial_navigation = None
        if start_url:
            normalized_start, start_error = _normalize_url(start_url)
            if start_error:
                return {
                    "ok": False,
                    "status": "blocked",
                    "backend": OPEN_SANDBOX_BACKEND,
                    "error": start_error,
                }
            assert normalized_start is not None
            start_url = normalized_start
            if is_placeholder_url(start_url) or is_unresolved_search_url(start_url):
                initial_navigation = resolve_initial_navigation(user_prompt or "Search the web")
                start_url = initial_navigation.resolved_url
        elif user_prompt:
            initial_navigation = resolve_initial_navigation(user_prompt)
            start_url = initial_navigation.resolved_url

        if start_url and urlsplit(start_url).scheme in {"http", "https"} and not _host_allowed(
            start_url, normalized_allowed_domains
        ):
            return {
                "ok": False,
                "status": "blocked",
                "backend": OPEN_SANDBOX_BACKEND,
                "error": "Navigation blocked: host is not in this task's domain allowlist.",
            }

        available = await self.availability()
        if not available.get("available"):
            return {
                "ok": False,
                "status": "unavailable",
                "backend": OPEN_SANDBOX_BACKEND,
                "error": available.get("error") or "OpenSandbox is unavailable.",
                "setup": (
                    "Start OpenSandbox server and configure THERIAD_OPENSANDBOX_DOMAIN; "
                    "the default is localhost:8080."
                ),
            }

        sandbox = None
        pw = None
        browser = None
        context = None
        page = None
        try:
            sandbox = await self._create_sandbox()
            endpoint = await sandbox.get_endpoint(DEFAULT_BROWSER_PORT)
            started_playwright = self._playwright().start()
            pw = (
                await started_playwright
                if inspect.isawaitable(started_playwright)
                else started_playwright
            )
            cdp_url = _cdp_url(endpoint)
            endpoint_headers = _endpoint_headers(endpoint)
            if endpoint_headers:
                browser = await pw.chromium.connect_over_cdp(
                    cdp_url,
                    headers=endpoint_headers,
                )
            else:
                browser = await pw.chromium.connect_over_cdp(cdp_url)
            contexts = list(browser.contexts)
            context = contexts[0] if contexts else await browser.new_context(
                viewport=viewport or {"width": 1280, "height": 900}
            )
            pages = list(context.pages)
            page = pages[0] if pages else await context.new_page()
            if viewport and hasattr(page, "set_viewport_size"):
                await page.set_viewport_size(viewport)
        except Exception as exc:
            await self._cleanup_handles(
                sandbox=sandbox,
                playwright=pw,
                browser=browser,
                context=context,
                page=page,
            )
            return {
                "ok": False,
                "status": "failed",
                "backend": OPEN_SANDBOX_BACKEND,
                "error": f"OpenSandbox browser could not start: {_redact_error(exc)}",
            }

        session = OpenSandboxBrowserSession(
            session_id=sid,
            sandbox=sandbox,
            playwright=pw,
            browser=browser,
            context=context,
            page=page,
            visible=visible,
            created_at=time.time(),
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
            allowed_domains=normalized_allowed_domains,
            user_prompt=user_prompt,
            current_url=getattr(page, "url", None),
            resolved_initial_url=start_url,
            navigation_reason=initial_navigation.reason if initial_navigation else None,
        )
        async with self._lock:
            self.sessions[sid] = session
        self._log(session, "start", "executed")

        if start_url:
            result = await self.navigate(
                sid,
                start_url,
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
            )
            if not result.get("ok"):
                await self.stop_session(
                    sid,
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
                return result

        return {
            "ok": True,
            "status": session.status,
            "backend": OPEN_SANDBOX_BACKEND,
            "browser": "chromium",
            "session_id": sid,
            "sandbox_id": getattr(sandbox, "id", None),
            "visible": visible,
            "current_url": _redact_text(session.current_url),
            "original_user_prompt": _redact_text(user_prompt),
            "resolved_initial_url": (
                _redact_text(session.resolved_initial_url)
                if session.resolved_initial_url
                else None
            ),
            "navigation_reason": (
                _redact_text(session.navigation_reason)
                if session.navigation_reason
                else None
            ),
            "allowed_domains": list(session.allowed_domains),
        }

    async def _cleanup_handles(
        self,
        *,
        sandbox: Any = None,
        playwright: Any = None,
        browser: Any = None,
        context: Any = None,
        page: Any = None,
    ) -> None:
        """Release local Playwright handles before destroying the sandbox.

        OpenSandbox owns the remote Chromium process.  The page/context/browser
        handles still own local protocol resources, so all layers are closed in
        order. Cleanup is best-effort and idempotent: a failure at one layer
        must not leave the remaining layers or the remote sandbox alive.
        """

        seen: set[int] = set()

        async def call(handle: Any, *methods: str) -> None:
            if handle is None or id(handle) in seen:
                return
            seen.add(id(handle))
            for method in methods:
                operation = getattr(handle, method, None)
                if operation is None:
                    continue
                try:
                    result = operation()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    # Continue through the remaining ownership layers.
                    pass
                return

        await call(page, "close")
        await call(context, "close")
        await call(browser, "close", "disconnect")
        await call(playwright, "stop")
        await call(sandbox, "destroy", "kill")

    async def _release_session(self, session: OpenSandboxBrowserSession) -> None:
        await self._cleanup_handles(
            sandbox=session.sandbox,
            playwright=session.playwright,
            browser=session.browser,
            context=session.context,
            page=session.page,
        )
        session.sandbox = None
        session.playwright = None
        session.browser = None
        session.context = None
        session.page = None

    async def _failed_session(
        self,
        session: OpenSandboxBrowserSession,
        action: str,
        error: str,
        *,
        target: str | None = None,
    ) -> dict[str, Any]:
        session.status = "failed"
        session.error = error
        self._log(session, action, "failed", target=target, error=error)
        await self._release_session(session)
        return self._snapshot(session, ok=False, error=error)

    async def stop_session(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            session = self._session(
                session_id,
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
            )
        except KeyError:
            return {"ok": False, "status": "not_found", "error": "Browser session not found."}
        session.status = "stopped"
        session.final_result = "Browser session stopped."
        self._log(session, "stop", "executed")
        await self._release_session(session)
        return self._snapshot(session)

    async def stop_all(self) -> None:
        for session_id in list(self.sessions):
            session = self.sessions.get(session_id)
            if session is not None:
                await self.stop_session(
                    session_id,
                    owner_user_id=session.owner_user_id,
                    owner_organization_id=session.owner_organization_id,
                )

    async def navigate(
        self,
        session_id: str,
        url: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        if session.page is None:
            return {"ok": False, "status": "stopped", "error": "Browser session is stopped."}
        requested_url = url
        normalized, error = _normalize_url(url)
        if error:
            self._log(session, "navigate", "blocked", target=requested_url, error=error)
            return self._snapshot(session, ok=False, error=error)
        assert normalized is not None

        if is_placeholder_url(normalized) or is_unresolved_search_url(normalized):
            navigation = resolve_initial_navigation(session.user_prompt or "Search the web")
            session.navigation_reason = navigation.reason
            session.resolved_initial_url = navigation.resolved_url
            normalized = navigation.resolved_url

        if urlsplit(normalized).scheme in {"http", "https"} and not _host_allowed(
            normalized, session.allowed_domains
        ):
            error = "Navigation blocked: host is not in this task's domain allowlist."
            self._log(session, "navigate", "blocked", target=requested_url, error=error)
            return self._snapshot(session, ok=False, error=error)

        try:
            await session.page.goto(normalized, wait_until="domcontentloaded", timeout=20_000)
        except Exception as exc:
            error = f"Navigation failed: {_redact_error(exc)}"
            return await self._failed_session(
                session, "navigate", error, target=normalized
            )
        session.current_url = getattr(session.page, "url", normalized)
        if (
            session.current_url
            and urlsplit(session.current_url).scheme in {"http", "https"}
            and not _host_allowed(session.current_url, session.allowed_domains)
        ):
            error = "Navigation blocked: redirect host is not in this task's domain allowlist."
            return await self._failed_session(
                session, "navigate", error, target=session.current_url
            )
        session.resolved_initial_url = session.resolved_initial_url or session.current_url
        self._log(session, "navigate", "executed", target=normalized)
        return await self.observe(
            session_id,
            screenshot=True,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = True,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        if session.page is None:
            return self._snapshot(session)
        try:
            dom = await session.page.evaluate(
                """
                (limit) => {
                  const visible = (el) => {
                    const style = getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.visibility !== 'hidden' && style.display !== 'none'
                      && rect.width > 0 && rect.height > 0;
                  };
                  const nodes = [...document.querySelectorAll(
                    'a[href], button, [role="button"], input, textarea, select'
                  )].filter(visible).slice(0, limit);
                  const elements = nodes.map((el, index) => {
                    let id = el.getAttribute('data-theriad-id');
                    if (!id) {
                      id = `osb-${index + 1}`;
                      el.setAttribute('data-theriad-id', id);
                    }
                    const tag = el.tagName.toLowerCase();
                    const type = (el.type || el.getAttribute('type') || '').toLowerCase();
                    // Never return live input values. Even non-password fields
                    // can contain credentials, one-time codes, or private data.
                    const text = (el.innerText || el.textContent || '')
                      .trim().slice(0, 160);
                    return {
                      id, tag, type, text,
                      label: el.getAttribute('aria-label') || '',
                      placeholder: el.getAttribute('placeholder') || '',
                      href: el.href || '',
                      in_form: Boolean(el.form),
                      download: el.hasAttribute('download')
                    };
                  });
                  return {
                    visible_text: (document.body?.innerText || '').trim().slice(0, 4000),
                    elements,
                    title: document.title || '',
                  };
                }
                """,
                MAX_ELEMENTS,
            )
            private_values = tuple(
                value for value in session.private_input_values if len(value) >= 3
            )
            visible_text = _redact_text(
                str(dom.get("visible_text") or ""),
                private_values=private_values,
            )[:MAX_VISIBLE_TEXT]
            session.current_url = getattr(session.page, "url", session.current_url)
            session.page_title = _redact_text(
                str(dom.get("title") or ""), private_values=private_values
            )[:300]
            session.page_summary = visible_text
            if screenshot:
                # A screenshot after typing can reveal the value even when the
                # DOM is redacted. Keep screenshots disabled for that session.
                if session.private_input_values:
                    session.screenshot_base64 = None
                else:
                    shot = await session.page.screenshot(type="png", full_page=False)
                    if len(shot) <= MAX_SCREENSHOT_BYTES:
                        session.screenshot_base64 = base64.b64encode(shot).decode("ascii")
            if self._is_captcha(session.current_url, session.page_title, visible_text):
                session.captcha_detected = True
                session.captcha_reason = (
                    "CAPTCHA requires a human to complete it in the sandbox browser."
                )
                session.status = "waiting_for_approval"
                session.pending_action = {
                    "action": "captcha_handoff",
                    "current_url": _redact_text(session.current_url),
                    "page_title": _redact_text(session.page_title),
                    "reason": session.captcha_reason,
                }
                session.pending_action_private = dict(session.pending_action)
                session.final_result = (
                    "CAPTCHA encountered. Complete it in the OpenSandbox browser or switch to API search."
                )
            elif session.status not in {"stopped", "completed"}:
                session.captcha_detected = False
                session.captcha_reason = None
                session.status = "running"
            elements = _safe_observation_elements(
                dom.get("elements", []), private_values=private_values
            )
            session.elements = elements
            return self._snapshot(session, extra={"elements": elements})
        except Exception as exc:
            error = f"Observation failed: {_redact_error(exc)}"
            return await self._failed_session(session, "observe", error)

    async def action(
        self,
        session_id: str,
        action: dict[str, Any] | None = None,
        *,
        approved: bool = False,
        deny: bool = False,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        if deny:
            pending = session.pending_action
            if pending is None:
                return self._snapshot(session, ok=False, error="No browser action is pending.")
            pending_name = _action_name(session.pending_action_private or pending)
            session.pending_action = None
            session.pending_action_private = None
            session.captcha_detected = False
            session.captcha_reason = None
            # A denied CAPTCHA handoff cannot safely make progress. Keep a
            # terminal state so an autonomous loop cannot rediscover and
            # re-prompt for the same challenge indefinitely.
            session.status = "failed" if pending_name == "captcha_handoff" else "running"
            session.final_result = (
                "CAPTCHA handoff denied; browser session cannot continue."
                if pending_name == "captcha_handoff"
                else session.final_result
            )
            self._log(
                session,
                pending_name,
                "denied",
            )
            return self._snapshot(session)

        pending = session.pending_action
        if pending:
            pending_private = session.pending_action_private or pending
            pending_name = _action_name(pending_private)
            if action is not None and _action_name(action) != pending_name:
                return self._snapshot(
                    session,
                    ok=False,
                    error=(
                        f"Browser action '{pending_name}' is awaiting resolution; "
                        "a different action cannot run first."
                    ),
                )
            if not approved:
                return self._snapshot(
                    session,
                    ok=False,
                    error="Browser action requires explicit approval or denial.",
                )
            action = dict(pending_private)
        action = dict(action or {})
        name = _action_name(action)

        if name == "captcha_handoff":
            if not approved:
                return self._snapshot(session, ok=False, error="CAPTCHA requires human approval or API search.")
            session.pending_action = None
            session.pending_action_private = None
            session.captcha_detected = False
            session.captcha_reason = None
            session.final_result = None
            return await self.observe(
                session_id,
                screenshot=True,
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
            )

        decision = self.policy.classify(name, approved=approved)
        if decision.status == "blocked":
            self._log(session, name, "blocked", error=decision.reason)
            return self._snapshot(session, ok=False, error=decision.reason)
        if decision.status == "approval_required" and not approved:
            session.status = "waiting_for_approval"
            session.pending_action_private = dict(action)
            session.pending_action = _safe_action_payload(action)
            session.pending_action["action"] = name
            session.pending_action["reason"] = decision.reason
            self._log(session, name, "approval_required", details=_safe_action_payload(action))
            return self._snapshot(session)

        if session.page is None:
            return self._snapshot(session, ok=False, error="Browser session is stopped.")

        try:
            if name == "navigate":
                return await self.navigate(
                    session_id,
                    str(action.get("url") or ""),
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
            if name in {"observe", "read_page", "screenshot"}:
                return await self.observe(
                    session_id,
                    screenshot=name == "screenshot" or bool(action.get("screenshot", True)),
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
            if name in {"click", "click_link", "click_button", "submit", "submit_form"}:
                locator = self._locator(session, action)
                if locator is None:
                    raise ValueError("element_id or selector is required for click actions")
                await locator.click(timeout=20_000)
            elif name in {"type", "fill"}:
                locator = self._locator(session, action)
                if locator is None:
                    raise ValueError("element_id or selector is required for typing actions")
                input_value = str(action.get("text") or action.get("value") or "")
                if input_value:
                    session.private_input_values.append(input_value)
                    del session.private_input_values[:-20]
                await locator.fill(input_value)
            elif name in {"press_key", "key"}:
                await session.page.keyboard.press(str(action.get("key") or "Enter"))
            elif name in {"scroll", "scroll_page"}:
                direction = str(action.get("direction") or "down").lower()
                amount = max(1, min(abs(int(action.get("amount") or 600)), 2_000))
                delta = -amount if direction in {"up", "left"} else amount
                await session.page.mouse.wheel(0, delta)
            elif name == "go_back":
                await session.page.go_back(wait_until="domcontentloaded", timeout=20_000)
            elif name == "go_forward":
                await session.page.go_forward(wait_until="domcontentloaded", timeout=20_000)
            elif name == "wait":
                milliseconds = max(0, min(int(action.get("milliseconds") or 500), 10_000))
                await session.page.wait_for_timeout(milliseconds)
            else:
                raise ValueError(f"Unsupported browser action '{name}'.")
            session.current_url = getattr(session.page, "url", session.current_url)
            session.pending_action = None
            session.pending_action_private = None
            session.status = "running"
            self._log(session, name, "executed", details=_safe_action_payload(action))
            return await self.observe(
                session_id,
                screenshot=True,
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
            )
        except Exception as exc:
            error = f"Action '{name}' failed: {_redact_error(exc)}"
            return await self._failed_session(
                session, name, error, target=str(action.get("element_id") or action.get("selector") or "")
            )

    @staticmethod
    def _locator(session: OpenSandboxBrowserSession, action: dict[str, Any]) -> Any:
        element_id = action.get("element_id") or action.get("selector")
        if not element_id:
            return None
        value = str(element_id)
        if value.startswith(("osb-", "ae-")):
            return session.page.locator(f'[data-theriad-id="{value}"]')
        return session.page.locator(value)

    async def use_api_search(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        from theriad.search import SearchService

        try:
            results = await SearchService().search(session.user_prompt)
            session.artifacts = [
                {"type": "search_result", **result.to_dict()} for result in results
            ]
            session.pending_action = None
            session.pending_action_private = None
            session.captcha_detected = False
            session.captcha_reason = None
            session.status = "completed"
            if results:
                session.final_result = "API search results: " + "; ".join(
                    f"{result.title}: {result.url}" for result in results[:5] if result.url
                )
            else:
                session.final_result = "API search returned no results."
            self._log(session, "use_api_search", "executed")
            await self._release_session(session)
            return self._snapshot(session)
        except Exception as exc:
            error = f"API search failed: {_redact_error(exc)}"
            return await self._failed_session(session, "use_api_search", error)

    async def status(
        self,
        session_id: str | None = None,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        if session_id:
            try:
                session = self._session(
                    session_id,
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
            except KeyError:
                return {"ok": False, "status": "not_found", "error": "Browser session not found."}
            return self._snapshot(session)
        available = await self.availability()
        return {
            "ok": bool(available.get("available")),
            "status": "available" if available.get("available") else "unavailable",
            "backend": OPEN_SANDBOX_BACKEND,
            "browser": "chromium",
            "active_sessions": len(self.sessions),
            **available,
        }

    def _snapshot(
        self,
        session: OpenSandboxBrowserSession,
        *,
        ok: bool = True,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session.current_url = session.current_url or (
            getattr(session.page, "url", None) if session.page is not None else None
        )
        private_values = tuple(
            value for value in session.private_input_values if len(value) >= 3
        )
        safe_extra = extra or {}
        elements = _safe_observation_elements(
            safe_extra.get("elements", session.elements), private_values=private_values
        )
        payload: dict[str, Any] = {
            "ok": ok,
            "status": session.status,
            "backend": OPEN_SANDBOX_BACKEND,
            "browser_backend": OPEN_SANDBOX_BACKEND,
            "observation_backend": OPEN_SANDBOX_BACKEND,
            "action_backend": OPEN_SANDBOX_BACKEND,
            "browser": "chromium",
            "session_id": session.session_id,
            "sandbox_id": getattr(session.sandbox, "id", None),
            "visible": session.visible,
            "current_url": _redact_text(session.current_url),
            "page_title": _redact_text(session.page_title, private_values=private_values),
            "current_page_summary": _redact_text(
                session.page_summary, private_values=private_values
            ),
            "visible_text_summary": _redact_text(
                session.page_summary, private_values=private_values
            ),
            "screenshot_base64": session.screenshot_base64,
            "pending_action": session.pending_action,
            "pending_approval": session.pending_action,
            "final_result": _redact_text(
                session.final_result, private_values=private_values
            )
            if session.final_result
            else None,
            "final_answer": _redact_text(
                session.final_result, private_values=private_values
            )
            if session.final_result
            else None,
            "error": _redact_text(error or session.error, private_values=private_values)
            if error or session.error
            else None,
            "captcha_detected": session.captcha_detected,
            "captcha_reason": _redact_text(session.captcha_reason),
            "original_user_prompt": _redact_text(
                session.user_prompt, private_values=private_values
            ),
            "resolved_initial_url": _redact_text(session.resolved_initial_url),
            "navigation_reason": _redact_text(session.navigation_reason),
            "allowed_domains": list(session.allowed_domains),
            "action_log": list(session.action_log),
            "artifacts": list(session.artifacts),
            "elements": elements,
            "clickable_elements": [
                item
                for item in elements
                if item.get("tag") in {"a", "button"}
                or item.get("role") == "button"
            ],
            "input_fields": [
                item for item in elements if item.get("tag") in {"input", "textarea", "select"}
            ],
            "accessibility_tree_summary": [
                {
                    "name": item.get("label") or item.get("text") or item.get("placeholder", ""),
                    "role": item.get("tag", ""),
                }
                for item in elements
            ],
            "elapsed_seconds": round(max(0.0, time.time() - session.created_at), 3),
        }
        if extra:
            payload.update({key: value for key, value in extra.items() if key != "elements"})
        return payload


_runtime = OpenSandboxBrowserRuntime()


def get_opensandbox_browser_runtime() -> OpenSandboxBrowserRuntime:
    return _runtime
