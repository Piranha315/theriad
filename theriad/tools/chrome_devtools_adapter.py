"""
Chrome DevTools MCP adapter.

Wraps the Chrome DevTools MCP server as an Theriad tool backend.

Availability
------------
The adapter is considered available when:
  1. THERIAD_CHROME_MCP_ENABLED=true (or legacy CHROME_MCP_ENABLED=true), OR
  2. ``CHROME_MCP_WS_URL`` is set to a running CDP/DevTools WebSocket endpoint.

If neither is set the adapter reports ``available=False`` and every call
returns an error immediately — the rest of the application continues normally.

Runtime Profile
---------------
The adapter reads the active RuntimeConfig (via get_runtime_config()) on
construction and exposes profile-related attributes used by the status endpoint:

    browser_profile_mode  — ephemeral | dedicated_persistent | real_profile
    allow_real_chrome_profile — bool
    headless              — bool
    runtime_profile       — str (value of RuntimeProfile enum)

consumer_secure (the default) always forces:
    browser_profile_mode = ephemeral
    allow_real_chrome_profile = False

Security
--------
- Default (consumer_secure) always uses an isolated ephemeral browser profile.
- real_profile may only be used when runtime_profile=personal_trusted AND
  allow_real_chrome_profile=True.
- Never receives provider API keys, local secrets, cookies, or env vars.
- Webpage content is treated as untrusted; page instructions cannot change policy.
- Sensitive actions (fill, submit_form, upload_file) are blocked by the broker
  before they reach this adapter.

Transport
---------
MCP invocation uses JSON-RPC over subprocess stdio.  One client is retained
for the lifetime of this adapter so a multi-step browser session keeps the
same Chrome MCP process and page state.  ``close()`` releases that transport.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any

from theriad.runtime.profile import get_runtime_config

logger = logging.getLogger(__name__)

# Tools exposed by Chrome DevTools MCP that we know about.
# This list is informational — the broker's allowlists govern what gets called.
_CHROME_MCP_TOOLS = [
    "new_page",
    "navigate_page",
    "take_screenshot",
    "take_snapshot",
    "evaluate_script",
    "list_console_messages",
    "list_network_requests",
    "wait_for",
    "click",
    "fill",
    "submit_form",
    "upload_file",
]

# Theriad's compatibility facade historically called the accessibility
# snapshot tool ``snapshot``. Current chrome-devtools-mcp calls it
# ``take_snapshot`` and uses page-scoped UIDs rather than CSS selectors.
_LEGACY_TOOL_ALIASES = {
    "snapshot": "take_snapshot",
}


def _javascript_function(source: str) -> str:
    """Turn a legacy expression/script into current evaluate_script input."""
    code = str(source or "").strip()
    if not code:
        return "() => undefined"
    if "=>" in code or code.startswith("function"):
        return code
    if ";" in code or "return " in code:
        return f"() => {{ {code} }}"
    return f"() => ({code})"


def _translate_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Translate Theriad's legacy Chrome shape to current MCP parameters."""
    actual_tool = _LEGACY_TOOL_ALIASES.get(tool_name, tool_name)
    translated = dict(arguments or {})

    if actual_tool == "new_page":
        translated.setdefault("url", "about:blank")
    elif actual_tool in {"click", "fill"}:
        if "uid" not in translated and "selector" in translated:
            translated["uid"] = translated.pop("selector")
    elif actual_tool == "evaluate_script":
        if "function" not in translated:
            source = translated.pop("expression", None)
            if source is None:
                source = translated.pop("script", None)
            if source is not None:
                translated["function"] = _javascript_function(str(source))

    # wait_for used to mean "let the page settle" and accepted only a
    # timeout. The current tool requires text. A snapshot is the closest
    # bounded readiness probe when no text was supplied.
    if actual_tool == "wait_for" and not translated.get("text"):
        actual_tool = "take_snapshot"
        translated.pop("timeout", None)

    page_id = translated.get("pageId")
    if isinstance(page_id, str) and page_id.isdigit():
        translated["pageId"] = int(page_id)
    return actual_tool, translated


def _with_page_id(result: dict[str, Any]) -> dict[str, Any]:
    """Expose the current page id when MCP returns it in page-list metadata."""
    if result.get("pageId") is not None or result.get("page_id") is not None:
        return result
    pages = result.get("pages") or result.get("tabs") or []
    if not isinstance(pages, list) or not pages:
        return result
    latest_page = pages[-1]
    if not isinstance(latest_page, dict):
        return result
    raw_page_id = (
        latest_page.get("pageId")
        if latest_page.get("pageId") is not None
        else latest_page.get("page_id")
    )
    if raw_page_id is None:
        raw_page_id = latest_page.get("id")
    if raw_page_id is None:
        return result
    page_id_text = str(raw_page_id).strip()
    page_id: int | str = int(page_id_text) if page_id_text.isdigit() else page_id_text
    return {**result, "pageId": page_id}


def _normalize_mcp_result(result: Any) -> dict[str, Any]:
    """Expose useful fields from MCP content blocks to legacy callers."""
    if not isinstance(result, dict):
        return {"value": result}

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return _with_page_id({**result, **structured, "value": structured})

    content = result.get("content")
    if not isinstance(content, list):
        return _with_page_id(result)

    images = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
    if images:
        first = images[0]
        return _with_page_id({
            **result,
            "screenshot": first.get("data"),
            "mime_type": first.get("mimeType"),
        })

    text = "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ).strip()
    if not text:
        return _with_page_id(result)
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, dict):
        return _with_page_id({**result, **decoded, "value": decoded})
    if decoded is None:
        page_match = re.search(r"\bpage(?:\s+id)?\s*[:#]?\s*(\d+)\b", text, re.IGNORECASE)
        return _with_page_id({
            **result,
            "value": text,
            **({"pageId": int(page_match.group(1))} if page_match else {}),
        })
    return _with_page_id({**result, "value": decoded})

# Env vars that signal Chrome MCP is configured (legacy names still honoured)
_ENV_ENABLED        = "THERIAD_CHROME_MCP_ENABLED"
_ENV_ENABLED_LEGACY = "CHROME_MCP_ENABLED"
_ENV_WS_URL         = "CHROME_MCP_WS_URL"


def _is_configured() -> tuple[bool, str | None]:
    """
    Return (configured: bool, ws_url_or_None).

    Checks environment variables only — no subprocess spawn or network I/O.
    New THERIAD_CHROME_MCP_ENABLED takes priority over legacy CHROME_MCP_ENABLED.
    """
    ws_url = os.environ.get(_ENV_WS_URL, "").strip()

    # New env var first, then legacy.  An explicit "false" must override a
    # stale legacy "true" rather than being treated as a missing value.
    enabled_raw = os.environ.get(_ENV_ENABLED)
    if enabled_raw is None:
        enabled_raw = os.environ.get(_ENV_ENABLED_LEGACY, "")
    enabled_raw = enabled_raw.strip().lower()

    if ws_url:
        return True, ws_url
    if enabled_raw in ("1", "true", "yes"):
        # The boolean flag asks chrome-devtools-mcp to launch its own
        # isolated browser.  Do not synthesize a WebSocket URL: /json is a
        # DevTools discovery endpoint, not a browser WebSocket endpoint.
        return True, None
    return False, None


class ChromeDevToolsAdapter:
    """
    Adapter for the Chrome DevTools MCP server.

    Attributes
    ----------
    available : bool
        True when the MCP server is configured and believed reachable.
    connected : bool
        True after a successful health-check (not yet implemented; mirrors
        ``available`` until real transport is wired).
    error : str | None
        Human-readable reason when ``available`` is False.
    tools : list[str]
        Known tool names exposed by this server.
    runtime_profile : str
        Active RuntimeProfile value ("personal_trusted" | "consumer_secure").
    browser_profile_mode : str
        Active BrowserProfileMode value.
    allow_real_chrome_profile : bool
        Whether real_profile is allowed under the current runtime config.
    headless : bool
        Whether Chrome should run headless.
    chrome_user_data_dir : str | None
        Effective user data dir path, or None for ephemeral.
    """

    def __init__(self, runtime_config: Any | None = None) -> None:
        configured, ws_url = _is_configured()
        self._ws_url = ws_url
        self._client: Any | None = None
        self._client_lock = threading.RLock()

        # Load runtime config (reads THERIAD_* env vars; default=consumer_secure)
        rc = runtime_config or get_runtime_config()
        self._runtime_config = rc

        # Expose profile-related attributes for status endpoint and tests
        self.runtime_profile:          str       = rc.runtime_profile.value
        self.browser_profile_mode:     str       = rc.browser_profile_mode.value
        self.allow_real_chrome_profile: bool     = rc.allow_real_chrome_profile
        self.headless:                 bool      = rc.chrome_headless
        self.chrome_user_data_dir:     str | None = rc.effective_chrome_user_data_dir

        self.tools:     list[str]  = _CHROME_MCP_TOOLS if configured else []
        self.available: bool       = configured
        self.connected: bool       = configured   # will differ once real ping is added
        self.error:     str | None = (
            None if configured
            else (
                f"Chrome DevTools MCP is not configured. "
                f"Set {_ENV_ENABLED}=true or {_ENV_WS_URL}=<ws-url> to enable."
            )
        )

        if configured:
            logger.info(
                "[chrome_devtools_adapter] configured ws_url=%s profile=%s browser_mode=%s",
                ws_url,
                self.runtime_profile,
                self.browser_profile_mode,
            )
        else:
            logger.debug("[chrome_devtools_adapter] not configured — skipping")

    # ── Public interface (called only by MCPBroker) ───────────────────────────

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke a Chrome DevTools MCP tool.

        The broker has already validated the allowlist and budget before
        calling this method. This method must not perform its own policy
        checks — that is the broker's responsibility.

        Returns a dict that will be placed in ``MCPCallResult.result``.

        Raises
        ------
        RuntimeError
            When the adapter is not available (broker should have checked).
        """
        if not self.available:
            raise RuntimeError(
                f"ChromeDevToolsAdapter is not available: {self.error}"
            )

        return self._dispatch(tool_name, arguments)

    def close(self) -> None:
        """Close the session-scoped MCP client, if it has been started."""
        with self._client_lock:
            self._close_client_locked()

    # ── Internal transport ────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Route a tool call to the Chrome DevTools MCP server via stdio.

        Reuses one chrome-devtools-mcp subprocess for this adapter, performing
        the MCP initialize handshake once and preserving page state across
        navigation, observation, and action calls. A transport error closes
        the client so the next call cannot reuse a poisoned process.

        Raises
        ------
        RuntimeError
            When the subprocess fails to start, the handshake times out, or
            the tool returns an error. The broker converts this into a blocked
            MCPCallResult so the caller always sees ok=False, never an exception.
        """
        from theriad.tools.mcp_stdio_client import MCPError, MCPStdioClient

        rc = self._runtime_config
        cmd = self._build_command(rc)
        actual_tool, translated_arguments = _translate_tool_call(tool_name, arguments)

        logger.info(
            "[chrome_devtools_adapter] dispatch tool=%s actual_tool=%s profile=%s browser_mode=%s",
            tool_name, actual_tool, self.runtime_profile, self.browser_profile_mode,
        )

        with self._client_lock:
            try:
                if self._client is None:
                    self._client = MCPStdioClient(
                        cmd,
                        startup_timeout=20.0,
                        default_timeout=30.0,
                    )
                result = _normalize_mcp_result(
                    self._client.call_tool(actual_tool, translated_arguments)
                )
                self.connected = True
                return result
            except MCPError as exc:
                self.connected = False
                self._close_client_locked()
                raise RuntimeError(f"Chrome MCP error calling '{tool_name}': {exc}") from exc
            except TimeoutError as exc:
                self.connected = False
                self._close_client_locked()
                raise RuntimeError(
                    f"Chrome MCP timed out calling '{tool_name}': {exc}"
                ) from exc
            except Exception:
                self.connected = False
                self._close_client_locked()
                raise

    def _close_client_locked(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.close()
        except Exception as exc:
            logger.debug("[chrome_devtools_adapter] client close failed: %s", exc)

    def _build_command(self, rc: Any) -> list[str]:
        """
        Build the subprocess argv for chrome-devtools-mcp.

        Flags chosen based on the explicit attach mode:
          - isolated: temporary profile, cleaned up by Chrome DevTools MCP
          - auto_connect: user-approved attach to an already-running Chrome
          - user_data_dir: dedicated non-default automation profile
          - --headless when chrome_headless=True
        """
        from theriad.runtime.profile import BrowserProfileMode

        # Start from the configured command + args
        cmd = [rc.chrome_mcp_command] + list(rc.chrome_mcp_args)

        # An explicit endpoint means the caller owns the Chrome process. Do
        # not also request --isolated/--headless, which would ask the MCP
        # server to launch a second browser instead of attaching to this one.
        if self._ws_url:
            if self._ws_url.startswith(("ws://", "wss://")):
                if not any(
                    arg in {"-w", "--wsEndpoint", "--ws-endpoint"}
                    or arg.startswith(("--wsEndpoint=", "--ws-endpoint="))
                    for arg in cmd
                ):
                    cmd.append(f"--wsEndpoint={self._ws_url}")
            elif not any(
                arg in {"-u", "--browserUrl", "--browser-url"}
                or arg.startswith(("--browserUrl=", "--browser-url="))
                for arg in cmd
            ):
                cmd.append(f"--browserUrl={self._ws_url}")
        else:
            attach_mode = getattr(rc, "chrome_mcp_attach_mode", "isolated")
            if attach_mode == "auto_connect":
                cmd.append("--autoConnect")
            elif attach_mode == "isolated" or rc.browser_profile_mode == BrowserProfileMode.ephemeral:
                cmd.append("--isolated")
            elif attach_mode == "user_data_dir" and rc.effective_chrome_user_data_dir:
                cmd.append(f"--userDataDir={rc.effective_chrome_user_data_dir}")

        # Keep credentials and private network metadata out of model-visible
        # tool results and disable optional usage telemetry for local tasks.
        if not any(arg.startswith("--redactNetworkHeaders") for arg in cmd):
            cmd.append("--redactNetworkHeaders=true")
        if "--no-usage-statistics" not in cmd:
            cmd.append("--no-usage-statistics")

        if rc.chrome_headless and not self._ws_url:
            cmd.append("--headless")

        return cmd

    # ── Health check (optional, cheap) ───────────────────────────────────────

    def ping(self) -> bool:
        """
        Attempt a lightweight connectivity check.

        Returns True if the MCP server appears reachable, False otherwise.
        Never raises.
        """
        if not self.available:
            return False
        # Future: try a GET to self._ws_url and check HTTP 200.
        # For now we optimistically return True when configured.
        return True
