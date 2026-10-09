"""
OpenRouter MCP adapter.

Launches ``npx openrouter-mcp`` as a subprocess and exposes the
``ask_model`` tool through MCPBroker so agents can consult any model
available on OpenRouter without an extra API surface.

Env vars
--------
OPENROUTER_API_KEY   Required.  Get one at https://openrouter.ai/keys.
OPENROUTER_MODEL     Optional default model for ask_model calls
                     (e.g. "anthropic/claude-3-haiku").  The tool also
                     accepts a ``model`` argument at call time.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.tools.mcp_stdio_client import MCPError, MCPStdioClient

logger = get_logger("tools.openrouter_mcp")

_MCP_COMMAND = ["npx", "openrouter-mcp"]


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        from theriad.settings.secrets import get_provider_api_key
        return get_provider_api_key("openrouter") or ""
    except Exception:
        return ""


class OpenRouterMCPAdapter:
    """
    MCPBroker-compatible adapter for the openrouter-mcp server.

    Exposes a single ``ask_model`` tool that routes a prompt through
    OpenRouter to any of its supported models.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client: MCPStdioClient | None = None
        self._error: str | None = None

        resolved_key = api_key or _get_api_key()
        if not resolved_key:
            self._error = (
                "No OpenRouter API key found. Add one in Settings → Providers or "
                "set the OPENROUTER_API_KEY environment variable. "
                "Get a key at https://openrouter.ai/keys."
            )
            return

        if not shutil.which("npx"):
            self._error = (
                "Node.js / npx is required to run the OpenRouter MCP server but was not found. "
                "Install Node.js (https://nodejs.org) and try again."
            )
            return

        env = {**os.environ, "OPENROUTER_API_KEY": resolved_key}
        default_model = os.environ.get("OPENROUTER_MODEL", "").strip()
        if default_model:
            env["OPENROUTER_MODEL"] = default_model

        try:
            self._client = MCPStdioClient(
                _MCP_COMMAND,
                startup_timeout=15.0,
                default_timeout=60.0,
                env=env,
            )
        except MCPError as exc:
            self._error = (
                f"Failed to start the OpenRouter MCP server: {exc.message}. "
                "Verify your OPENROUTER_API_KEY is valid and that npx can reach "
                "the internet to download the openrouter-mcp package."
            )
        except Exception as exc:
            self._error = (
                f"Unexpected error starting the OpenRouter MCP server: {exc}."
            )

    @property
    def available(self) -> bool:
        return self._client is not None and self._error is None

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(self._error or "OpenRouterMCPAdapter is not initialised.")
        return self._client.call_tool(tool_name, arguments)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OpenRouterMCPAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        if not shutil.which("npx"):
            return False, "Node.js / npx not found."
        if not _get_api_key():
            return False, "No OpenRouter API key configured."
        return True, None
