"""
LLM MCP adapter — wraps ``theriad/tools/llm_mcp_server.py`` as a subprocess.

Launches the built-in MCP server (llm_mcp_server.py) with provider-specific
environment variables so any OpenAI-compatible endpoint (Groq, DeepInfra, …)
can be called as an MCP tool via MCPBroker.

Pre-configured subclasses
--------------------------
    GroqMCPAdapter      — api.groq.com; requires GROQ_API_KEY or secrets.json
    DeepInfraMCPAdapter — api.deepinfra.com; requires DEEPINFRA_API_KEY or secrets.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from theriad.core.logging.logger import get_logger
from theriad.tools.mcp_stdio_client import MCPError, MCPStdioClient

logger = get_logger("tools.llm_mcp")

# Absolute path to the built-in server so the subprocess can always find it.
_SERVER_PATH = str(Path(__file__).with_name("llm_mcp_server.py"))

_GROQ_BASE_URL     = "https://api.groq.com/openai/v1"
_DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


class LLMMCPAdapter:
    """
    MCPBroker-compatible adapter for any OpenAI-compatible LLM provider.

    Launches ``llm_mcp_server.py`` as a subprocess and communicates over stdio.
    Exposes a single ``complete`` tool.

    Parameters
    ----------
    base_url:
        OpenAI-compatible base URL, e.g. "https://api.groq.com/openai/v1".
    api_key:
        Bearer token for the provider.
    model:
        Default model identifier.
    provider_name:
        Human-readable label used in the tool description (e.g. "Groq").
    timeout:
        Per-request LLM timeout in seconds (default 60).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider_name: str = "LLM",
        timeout: float = 60.0,
    ) -> None:
        self._client: MCPStdioClient | None = None
        self._error: str | None = None
        self._provider_name = provider_name

        if not base_url or not api_key or not model:
            self._error = (
                f"{provider_name}: base_url, api_key, and model are all required."
            )
            return

        env = {
            **os.environ,
            "LLM_MCP_BASE_URL": base_url,
            "LLM_MCP_API_KEY": api_key,
            "LLM_MCP_MODEL": model,
            "LLM_MCP_PROVIDER": provider_name,
            "LLM_MCP_TIMEOUT": str(timeout),
        }

        try:
            self._client = MCPStdioClient(
                [sys.executable, _SERVER_PATH],
                startup_timeout=10.0,
                default_timeout=timeout + 5.0,
                env=env,
            )
        except MCPError as exc:
            self._error = (
                f"Failed to start the {provider_name} MCP server: {exc.message}."
            )
        except Exception as exc:
            self._error = (
                f"Unexpected error starting the {provider_name} MCP server: {exc}."
            )

    @property
    def available(self) -> bool:
        return self._client is not None and self._error is None

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(
                self._error or f"{self._provider_name} MCPAdapter is not initialised."
            )
        return self._client.call_tool(tool_name, arguments)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LLMMCPAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _resolve_key(provider: str) -> str:
    """Resolve a provider key from secrets, returning '' if not found."""
    try:
        from theriad.settings.secrets import get_provider_api_key
        return get_provider_api_key(provider) or ""
    except Exception:
        return ""


class GroqMCPAdapter(LLMMCPAdapter):
    """
    Pre-configured LLMMCPAdapter for Groq.

    Reads the API key from GROQ_API_KEY env var or secrets.json.
    Defaults to llama-3.3-70b-versatile; pass ``model`` to override.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "").strip() or _resolve_key("groq")
        if not resolved_key:
            # Defer the "not configured" state to the base class
            super().__init__(
                base_url=_GROQ_BASE_URL,
                api_key="",
                model=model or self.DEFAULT_MODEL,
                provider_name="Groq",
            )
            self._error = (
                "No Groq API key found. Add one in Settings → Providers or "
                "set the GROQ_API_KEY environment variable. "
                "Get a key at https://console.groq.com/keys."
            )
            return
        super().__init__(
            base_url=_GROQ_BASE_URL,
            api_key=resolved_key,
            model=model or self.DEFAULT_MODEL,
            provider_name="Groq",
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        key = os.environ.get("GROQ_API_KEY", "").strip() or _resolve_key("groq")
        if not key:
            return False, "No Groq API key configured."
        return True, None


class DeepInfraMCPAdapter(LLMMCPAdapter):
    """
    Pre-configured LLMMCPAdapter for DeepInfra.

    Reads the API key from DEEPINFRA_API_KEY env var or secrets.json.
    Defaults to meta-llama/Llama-3.3-70B-Instruct; pass ``model`` to override.
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("DEEPINFRA_API_KEY", "").strip()
            or _resolve_key("deepinfra")
        )
        if not resolved_key:
            super().__init__(
                base_url=_DEEPINFRA_BASE_URL,
                api_key="",
                model=model or self.DEFAULT_MODEL,
                provider_name="DeepInfra",
            )
            self._error = (
                "No DeepInfra API key found. Add one in Settings → Providers or "
                "set the DEEPINFRA_API_KEY environment variable. "
                "Get a key at https://deepinfra.com/dash."
            )
            return
        super().__init__(
            base_url=_DEEPINFRA_BASE_URL,
            api_key=resolved_key,
            model=model or self.DEFAULT_MODEL,
            provider_name="DeepInfra",
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        key = (
            os.environ.get("DEEPINFRA_API_KEY", "").strip() or _resolve_key("deepinfra")
        )
        if not key:
            return False, "No DeepInfra API key configured."
        return True, None
