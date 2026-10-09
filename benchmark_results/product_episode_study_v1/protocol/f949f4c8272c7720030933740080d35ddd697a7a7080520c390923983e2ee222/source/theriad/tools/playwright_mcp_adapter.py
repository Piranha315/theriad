"""
Playwright MCP adapter.

Launches ``npx @playwright/mcp`` as a subprocess and wraps it with
MCPStdioClient so the MCPBroker can call browser tools the same way it
already calls Chrome DevTools MCP tools.

Backend selection
-----------------
Add the adapter to MCPBroker at construction time::

    adapter = PlaywrightMCPAdapter()
    broker  = MCPBroker(session_id="…", adapters={"playwright_mcp": adapter})

Availability
------------
Requires Node.js and the ``@playwright/mcp`` npm package.  The package is
downloaded automatically on first use via ``npx`` when not already cached.
Chromium must be installed (via ``playwright install chromium``).
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.tools.mcp_stdio_client import MCPError, MCPStdioClient

logger = get_logger("tools.playwright_mcp")

# The command printed by `npx @playwright/mcp --headless --isolated`.
# --isolated keeps the browser profile in memory (no disk footprint).
# --headless runs without a visible window.
_MCP_COMMAND = ["npx", "@playwright/mcp", "--headless", "--isolated"]


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npx") is not None


class PlaywrightMCPAdapter:
    """
    MCPBroker-compatible adapter for the official Playwright MCP server.

    One adapter instance corresponds to one browser session (one subprocess).
    Call close() when the session ends.
    """

    def __init__(self) -> None:
        self._client: MCPStdioClient | None = None
        self._error: str | None = None

        if not _node_available():
            self._error = (
                "Node.js is required to run the Playwright MCP server but was not found. "
                "Install Node.js (https://nodejs.org) and try again."
            )
            return

        # Check PLAYWRIGHT_BROWSERS_PATH so the subprocess inherits it.
        env: dict[str, str] = {}
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        if browsers_path:
            env["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
        skip_dl = os.environ.get("PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD")
        if skip_dl:
            env["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = skip_dl

        try:
            self._client = MCPStdioClient(
                _MCP_COMMAND,
                startup_timeout=20.0,
                default_timeout=30.0,
                env=env if env else None,
            )
        except MCPError as exc:
            self._error = (
                f"Failed to start the Playwright MCP server: {exc.message}. "
                "Ensure Node.js is installed and @playwright/mcp is accessible via npx. "
                "Run 'npx @playwright/mcp --version' to verify."
            )
        except Exception as exc:
            self._error = (
                f"Unexpected error starting the Playwright MCP server: {exc}. "
                "Check that Node.js and npx are correctly installed."
            )

    # -- MCPBroker interface --------------------------------------------------

    @property
    def available(self) -> bool:
        return self._client is not None and self._error is None

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(
                self._error or "PlaywrightMCPAdapter is not initialised."
            )
        return self._client.call_tool(tool_name, arguments)

    # -- Lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "PlaywrightMCPAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- Availability probe (class method, no subprocess started) -------------

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """
        Return (available, error_or_None) without starting a subprocess.

        Checks only that Node.js is present; does not verify that Chromium
        is installed.  Use the adapter's .available property for runtime
        confirmation.
        """
        if not _node_available():
            return False, (
                "Node.js / npx not found. Install Node.js to use the Playwright MCP backend."
            )
        return True, None
