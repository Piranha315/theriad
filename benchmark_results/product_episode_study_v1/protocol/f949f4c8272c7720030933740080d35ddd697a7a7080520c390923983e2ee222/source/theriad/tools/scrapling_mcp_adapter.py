"""
Scrapling MCP adapter.

Launches the Scrapling MCP server as a subprocess and wraps it with
MCPStdioClient so the MCPBroker can call stealth-fetch and search tools.

Installation
------------
pip install "scrapling[mcp]"

Once installed the server is available as::

    python -m scrapling.mcp

Tools exposed (once Scrapling ≥ 0.4 with MCP support is installed)
-------------------------------------------------------------------
- scrapling_fetch       — fetch a URL, basic mode (fast, no browser)
- scrapling_fetch_stealth — fetch via camoufox/Playwright stealth browser
- scrapling_search      — DuckDuckGo HTML scrape returning structured results
"""
from __future__ import annotations

import shutil
import sys
from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.tools.mcp_stdio_client import MCPError, MCPStdioClient

logger = get_logger("tools.scrapling_mcp")

_MCP_COMMAND = [sys.executable, "-m", "scrapling.mcp"]


def _scrapling_mcp_available() -> bool:
    try:
        import importlib
        importlib.import_module("scrapling.mcp")
        return True
    except ImportError:
        return False


class ScraplingMCPAdapter:
    """
    MCPBroker-compatible adapter for the Scrapling MCP server.

    Provides stealth web fetching and search as MCP tools.
    Requires: pip install "scrapling[mcp]"
    """

    def __init__(self) -> None:
        self._client: MCPStdioClient | None = None
        self._error: str | None = None

        if not _scrapling_mcp_available():
            self._error = (
                "The Scrapling MCP server is not installed. "
                "Run: pip install \"scrapling[mcp]\" to enable stealth web fetching as MCP tools."
            )
            return

        try:
            self._client = MCPStdioClient(
                _MCP_COMMAND,
                startup_timeout=15.0,
                default_timeout=30.0,
            )
        except MCPError as exc:
            self._error = (
                f"Failed to start the Scrapling MCP server: {exc.message}. "
                "Verify that scrapling[mcp] is correctly installed by running: "
                "python -m scrapling.mcp"
            )
        except Exception as exc:
            self._error = (
                f"Unexpected error starting the Scrapling MCP server: {exc}."
            )

    @property
    def available(self) -> bool:
        return self._client is not None and self._error is None

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(self._error or "ScraplingMCPAdapter is not initialised.")
        return self._client.call_tool(tool_name, arguments)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ScraplingMCPAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        if not _scrapling_mcp_available():
            return False, 'scrapling[mcp] not installed. Run: pip install "scrapling[mcp]"'
        return True, None
