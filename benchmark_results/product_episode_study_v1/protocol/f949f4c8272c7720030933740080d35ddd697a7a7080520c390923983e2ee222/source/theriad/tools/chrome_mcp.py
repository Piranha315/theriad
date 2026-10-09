"""
High-level Chrome MCP interface.

Provides a clean, method-per-action facade over MCPBroker / ChromeDevToolsAdapter.
Callers never interact with tool names or argument dicts directly.

Design rules:
- Dependency-injected broker; no global state.
- Every method returns a plain dict (the broker's MCPCallResult.result).
- Errors surface as ChromeMCPError rather than raw broker exceptions.
- Mock mode: when broker is None the methods return deterministic stubs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class ChromeMCPError(RuntimeError):
    """Raised when an MCP call fails or is blocked by policy."""


@dataclass
class ChromeMCP:
    """
    High-level Chrome MCP interface.

    Parameters
    ----------
    broker:
        An MCPBroker instance.  Pass ``None`` to run in stub/mock mode
        (useful in tests and when Chrome MCP is not configured).
    subtask_type:
        The active subtask type string forwarded to the broker for allowlist
        checks (e.g. ``"agent_safe_navigation"``).
    """

    broker: Any  # MCPBroker | None
    subtask_type: str = "agent_safe_navigation"
    _stub_url: str = field(default="about:blank", init=False, repr=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a call through the broker; raise ChromeMCPError on failure."""
        if self.broker is None:
            return self._stub(tool_name, arguments)

        result = self.broker.call(
            self.subtask_type,
            "chrome_devtools",
            tool_name,
            arguments,
        )
        if not result.ok:
            raise ChromeMCPError(
                f"MCP call '{tool_name}' failed: "
                f"{result.error or result.blocked_reason or 'blocked by policy'}"
            )
        return result.result if isinstance(result.result, dict) else {"value": result.result}

    def _stub(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic stub data when no broker is available."""
        if tool_name == "navigate_page":
            self._stub_url = arguments.get("url", self._stub_url)
            return {"url": self._stub_url, "title": "Stub Page", "ok": True}
        if tool_name == "take_screenshot":
            return {"screenshot": None, "format": "png", "stub": True}
        if tool_name == "snapshot":
            return {
                "url": self._stub_url,
                "title": "Stub Page",
                "visible_text": "Stub page content.",
                "links": [],
                "buttons": [],
            }
        if tool_name == "evaluate_script":
            code = arguments.get("expression", "")
            if "location.href" in code:
                return {"value": self._stub_url}
            if "history.back" in code:
                return {"value": None}
            if "history.forward" in code:
                return {"value": None}
            return {"value": None}
        if tool_name == "click":
            return {"clicked": arguments.get("selector", ""), "ok": True}
        if tool_name == "fill":
            return {"filled": arguments.get("selector", ""), "ok": True}
        if tool_name == "wait_for":
            return {"waited_ms": arguments.get("timeout", 500), "ok": True}
        if tool_name == "list_console_messages":
            return {"messages": []}
        return {"ok": True, "stub": True}

    # ── Public interface ──────────────────────────────────────────────────────

    def navigate(self, url: str) -> dict[str, Any]:
        """Navigate the active tab to *url*."""
        return self._call("navigate_page", {"url": url})

    def click(self, selector: str) -> dict[str, Any]:
        """Click the element matching *selector*."""
        return self._call("click", {"selector": selector})

    def type(self, selector: str, text: str) -> dict[str, Any]:
        """Fill *text* into the element matching *selector*."""
        return self._call("fill", {"selector": selector, "value": text})

    def read_dom(self) -> dict[str, Any]:
        """Return a structured snapshot of the current page DOM."""
        return self._call("snapshot", {})

    def screenshot(self) -> dict[str, Any]:
        """Capture a screenshot of the current page."""
        return self._call("take_screenshot", {})

    def current_url(self) -> str:
        """Return the URL of the active tab."""
        result = self._call("evaluate_script", {"expression": "window.location.href"})
        return str(result.get("value", "about:blank"))

    def back(self) -> dict[str, Any]:
        """Navigate back in the browser history."""
        return self._call("evaluate_script", {"expression": "history.back(); window.location.href"})

    def forward(self) -> dict[str, Any]:
        """Navigate forward in the browser history."""
        return self._call("evaluate_script", {"expression": "history.forward(); window.location.href"})

    def wait(self, timeout_ms: int = 500) -> dict[str, Any]:
        """Wait for *timeout_ms* milliseconds or until the page is stable."""
        return self._call("wait_for", {"timeout": timeout_ms})

    def evaluate_js(self, code: str) -> Any:
        """Evaluate arbitrary JavaScript and return the result."""
        result = self._call("evaluate_script", {"expression": code})
        return result.get("value")

    def list_tabs(self) -> list[dict[str, Any]]:
        """Return a list of open tabs as [{id, url, title}]."""
        result = self._call(
            "evaluate_script",
            {"expression": "JSON.stringify([{id: 0, url: window.location.href, title: document.title}])"},
        )
        value = result.get("value")
        if isinstance(value, list):
            return value
        # Parse JSON string when broker serialises the result
        import json
        try:
            return json.loads(value) if isinstance(value, str) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def switch_tab(self, tab_id: int) -> dict[str, Any]:
        """Switch to the tab identified by *tab_id*."""
        return self._call(
            "evaluate_script",
            {"expression": f"window.__switchTab && window.__switchTab({tab_id}); window.location.href"},
        )


def build_chrome_mcp(
    session_id: str,
    subtask_type: str = "agent_safe_navigation",
    *,
    broker: Optional[Any] = None,
) -> ChromeMCP:
    """
    Factory that wires a ChromeMCP to the active MCPBroker for *session_id*.

    If *broker* is supplied it is used directly (useful in tests).
    If Chrome MCP is not configured the returned instance runs in stub mode.
    """
    if broker is not None:
        return ChromeMCP(broker=broker, subtask_type=subtask_type)

    try:
        from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
        from theriad.tools.mcp_broker import MCPBroker
        from theriad.runtime.profile import get_runtime_config

        rc = get_runtime_config()
        adapter = ChromeDevToolsAdapter(runtime_config=rc)
        if not adapter.ping():
            raise ChromeMCPError(adapter.error or "Chrome DevTools MCP is not configured.")

        live_broker = MCPBroker(
            session_id=session_id,
            adapters={"chrome_devtools": adapter},
            runtime_config=rc,
        )
        return ChromeMCP(broker=live_broker, subtask_type=subtask_type)
    except Exception as exc:
        raise ChromeMCPError(f"Chrome DevTools MCP is unavailable: {exc}") from exc
