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

import json
import re
from dataclasses import dataclass, field
from typing import Any


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
    _page_id: int | str | None = field(default=None, init=False, repr=False)

    # Keep the broker's narrow policy boundaries intact even though this
    # facade exposes a method-per-action API.  The old implementation sent
    # every operation through ``agent_safe_navigation``; that allowlist does
    # not contain snapshot, screenshot, click, or fill.
    _operation_subtasks: dict[str, str] = field(
        default_factory=lambda: {
            "new_page": "agent_safe_navigation",
            "navigate_page": "agent_safe_navigation",
            "wait_for": "agent_safe_navigation",
            "snapshot": "agent_observe_dom",
            "evaluate_script": "agent_observe_dom",
            "take_screenshot": "agent_observe_screenshot",
            "click": "agent_safe_click",
            "fill": "agent_sensitive_action",
        },
        init=False,
        repr=False,
    )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _subtask_for(self, tool_name: str) -> str:
        """Choose the narrowest registered subtask for a Chrome operation."""
        # Preserve an explicitly supplied subtask when it already authorizes
        # this tool (notably website_preview_qa).  Otherwise use the
        # operation-specific compatibility allowlist.
        from theriad.tools.mcp_registry import get_allowlist

        active = get_allowlist(self.subtask_type)
        if active is not None and tool_name in active.allowed_tools:
            return self.subtask_type
        return self._operation_subtasks.get(tool_name, self.subtask_type)

    def _call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        subtask_type: str | None = None,
        approval_received: bool = False,
    ) -> dict[str, Any]:
        """Dispatch a call through the broker; raise ChromeMCPError on failure."""
        if self.broker is None:
            return self._stub(tool_name, arguments)

        call_arguments = dict(arguments)
        if self._page_id and tool_name != "new_page":
            call_arguments.setdefault("pageId", self._page_id)
        call_kwargs = {"approval_received": True} if approval_received else {}
        result = self.broker.call(
            subtask_type or self._subtask_for(tool_name),
            "chrome_devtools",
            tool_name,
            call_arguments,
            **call_kwargs,
        )
        if not result.ok:
            raise ChromeMCPError(
                f"MCP call '{tool_name}' failed: "
                f"{result.error or result.blocked_reason or 'blocked by policy'}"
            )
        raw_result = result.result if isinstance(result.result, dict) else {"value": result.result}
        return self._normalize_result(raw_result)

    @staticmethod
    def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
        """Flatten standard MCP content blocks into the legacy dict shape."""
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return {
                **result,
                **structured,
                "_mcp_content": result.get("content", []),
            }

        content = result.get("content")
        if not isinstance(content, list):
            return result

        images = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
        if images:
            first = images[0]
            return {
                **result,
                "screenshot": first.get("data"),
                "mime_type": first.get("mimeType"),
                "_mcp_content": content,
            }

        text = "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
        if not text:
            return result
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            return {**result, **decoded, "_mcp_content": content}
        return {**result, "value": decoded if decoded is not None else text, "_mcp_content": content}

    @staticmethod
    def _result_value(result: dict[str, Any]) -> Any:
        """Read scalar values across old and current MCP response shapes."""
        if "value" in result:
            return result["value"]
        nested = result.get("result")
        if isinstance(nested, dict) and "value" in nested:
            return nested["value"]
        return nested

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

    def new_page(self, url: str | None = None) -> dict[str, Any]:
        """Create or select the active Chrome MCP page."""
        result = self._call("new_page", {"url": url} if url else {})
        raw_page_id = (
            result.get("pageId")
            or result.get("page_id")
        )
        if raw_page_id is None:
            pages = result.get("pages") or result.get("tabs") or []
            if isinstance(pages, list) and pages:
                latest_page = pages[-1]
                if isinstance(latest_page, dict):
                    raw_page_id = (
                        latest_page.get("pageId")
                        or latest_page.get("page_id")
                        or latest_page.get("id")
                    )
        if raw_page_id is None:
            self._page_id = None
        else:
            page_id_text = str(raw_page_id).strip()
            try:
                self._page_id = int(page_id_text) if page_id_text.isdigit() else page_id_text
            except (TypeError, ValueError):
                self._page_id = None
            if not self._page_id:
                self._page_id = None
        return result

    def navigate(self, url: str) -> dict[str, Any]:
        """Navigate the active tab to *url*."""
        return self._call("navigate_page", {"url": url})

    def click(self, selector: str) -> dict[str, Any]:
        """Click the element matching *selector*."""
        return self._call("click", {"selector": selector})

    def type(
        self,
        selector: str,
        text: str,
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        """Fill *text* into the element matching *selector*."""
        return self._call(
            "fill",
            {"selector": selector, "value": text},
            approval_received=approved,
        )

    def close(self) -> None:
        """Close the underlying broker transport, if it owns one."""
        close = getattr(self.broker, "close", None)
        if callable(close):
            close()
        self._page_id = None

    def read_dom(self) -> dict[str, Any]:
        """Return a structured snapshot of the current page DOM."""
        result = self._call("snapshot", {})
        if result.get("visible_text"):
            return result
        raw_text = result.get("value") or result.get("text") or ""
        visible_text = str(raw_text)
        elements: list[dict[str, Any]] = []
        for match in re.finditer(
            r"(?im)^\s*-\s*(?P<role>[a-z][\w -]*)\s+[^\n]*?"
            r"(?P<label>\"[^\"]*\"|'[^']*'|[^\[]*?)?\s*"
            r"\[uid=(?P<uid>[^\]]+)\]",
            visible_text,
        ):
            label = str(match.group("label") or "").strip(" \"'")
            role = str(match.group("role") or "").strip().split()[0]
            elements.append({
                "id": match.group("uid"),
                "uid": match.group("uid"),
                "tag": role,
                "text": label,
            })
        return {
            **result,
            "visible_text": visible_text,
            "links": [item for item in elements if item.get("tag") in {"link", "a"}],
            "buttons": [item for item in elements if item.get("tag") in {"button", "link", "a"}],
            "raw_snapshot": visible_text,
        }

    def screenshot(self) -> dict[str, Any]:
        """Capture a screenshot of the current page."""
        return self._call("take_screenshot", {})

    def current_url(self) -> str:
        """Return the URL of the active tab."""
        result = self._call("evaluate_script", {"expression": "window.location.href"})
        return str(self._result_value(result) or "about:blank")

    def back(self) -> dict[str, Any]:
        """Navigate back in the browser history."""
        return self._call("navigate_page", {"type": "back"})

    def forward(self) -> dict[str, Any]:
        """Navigate forward in the browser history."""
        return self._call("navigate_page", {"type": "forward"})

    def wait(self, timeout_ms: int = 500) -> dict[str, Any]:
        """Wait for *timeout_ms* milliseconds or until the page is stable."""
        return self._call("wait_for", {"timeout": timeout_ms})

    def evaluate_js(self, code: str) -> Any:
        """Evaluate arbitrary JavaScript and return the result."""
        result = self._call("evaluate_script", {"expression": code})
        return self._result_value(result)

    def list_tabs(self) -> list[dict[str, Any]]:
        """Return a list of open tabs as [{id, url, title}]."""
        result = self._call(
            "evaluate_script",
            {"expression": "JSON.stringify([{id: 0, url: window.location.href, title: document.title}])"},
        )
        value = self._result_value(result)
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
    broker: Any | None = None,
) -> ChromeMCP:
    """
    Factory that wires a ChromeMCP to the active MCPBroker for *session_id*.

    If *broker* is supplied it is used directly (useful in tests).
    If Chrome MCP is not configured the returned instance runs in stub mode.
    """
    if broker is not None:
        return ChromeMCP(broker=broker, subtask_type=subtask_type)

    try:
        from theriad.runtime.profile import get_runtime_config
        from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
        from theriad.tools.mcp_broker import MCPBroker

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
