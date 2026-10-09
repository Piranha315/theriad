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

Transport (future)
------------------
Real MCP invocation will use JSON-RPC over a subprocess stdio or SSE
connection.  The ``_dispatch`` method is the single call-site; replace it
with actual transport when the MCP client is wired in.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from theriad.runtime.profile import BrowserProfileMode, RuntimeProfile, get_runtime_config

logger = logging.getLogger(__name__)

# Tools exposed by Chrome DevTools MCP that we know about.
# This list is informational — the broker's allowlists govern what gets called.
_CHROME_MCP_TOOLS = [
    "new_page",
    "navigate_page",
    "take_screenshot",
    "snapshot",
    "evaluate_script",
    "list_console_messages",
    "list_network_requests",
    "wait_for",
    "click",
    "fill",
    "submit_form",
    "upload_file",
]

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

    # New env var first, then legacy
    enabled_raw = (
        os.environ.get(_ENV_ENABLED, "")
        or os.environ.get(_ENV_ENABLED_LEGACY, "")
    ).strip().lower()

    if ws_url:
        return True, ws_url
    if enabled_raw in ("1", "true", "yes"):
        return True, "ws://localhost:9222/json"
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
        NotImplementedError
            When real MCP transport is not yet wired (current state).
        """
        if not self.available:
            raise RuntimeError(
                f"ChromeDevToolsAdapter is not available: {self.error}"
            )

        return self._dispatch(tool_name, arguments)

    # ── Internal transport ────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Route a tool call to the Chrome DevTools MCP server via stdio.

        Spawns a fresh chrome-devtools-mcp subprocess for each call using
        MCPStdioClient, performs the MCP initialize handshake, calls the tool,
        then terminates the subprocess.  This is intentionally stateless —
        one call, one subprocess — which keeps Preview QA isolated and
        avoids dangling processes on error.

        For performance-critical use cases (e.g. multi-step browser-agent
        sessions) a persistent client should be held in the session object
        and injected here. That optimisation is deferred to a later pass.

        Raises
        ------
        RuntimeError
            When the subprocess fails to start, the handshake times out, or
            the tool returns an error. The broker converts this into a blocked
            MCPCallResult so the caller always sees ok=False, never an exception.
        """
        from theriad.tools.mcp_stdio_client import MCPStdioClient, MCPError

        rc = self._runtime_config
        cmd = self._build_command(rc)

        logger.info(
            "[chrome_devtools_adapter] dispatch tool=%s profile=%s browser_mode=%s",
            tool_name, self.runtime_profile, self.browser_profile_mode,
        )

        try:
            with MCPStdioClient(cmd, startup_timeout=20.0, default_timeout=30.0) as client:
                return client.call_tool(tool_name, arguments)
        except MCPError as exc:
            raise RuntimeError(f"Chrome MCP error calling '{tool_name}': {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Chrome MCP timed out calling '{tool_name}': {exc}"
            ) from exc

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

        if rc.chrome_headless:
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
