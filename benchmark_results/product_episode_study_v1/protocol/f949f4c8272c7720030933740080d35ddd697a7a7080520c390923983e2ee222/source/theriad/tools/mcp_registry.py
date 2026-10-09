"""
Static tool capability registry and subtask allowlists.

TOOL_REGISTRY maps "server:tool" keys to MCPToolSpec entries with risk
metadata. SUBTASK_ALLOWLISTS maps subtask names to the narrow set of tools
that subtask is permitted to call — the broker enforces these at runtime.

Adding a new MCP server:
    1. Add its tools to TOOL_REGISTRY.
    2. Add (or extend) subtask entries in SUBTASK_ALLOWLISTS.
    3. Implement an adapter in theriad/tools/ and register it with MCPBroker.
"""
from __future__ import annotations

from theriad.tools.mcp_schemas import MCPToolSpec, RiskLevel, SubtaskAllowlist

# ── Tool registry ─────────────────────────────────────────────────────────────
# Key format: "<server_name>:<tool_name>"

TOOL_REGISTRY: dict[str, MCPToolSpec] = {
    # ── Chrome DevTools MCP — observation / read-only ─────────────────────
    "chrome_devtools:new_page": MCPToolSpec(
        server_name="chrome_devtools", tool_name="new_page",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "chrome_devtools:navigate_page": MCPToolSpec(
        server_name="chrome_devtools", tool_name="navigate_page",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=30, timeout_seconds=15.0,
    ),
    "chrome_devtools:take_screenshot": MCPToolSpec(
        server_name="chrome_devtools", tool_name="take_screenshot",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "chrome_devtools:snapshot": MCPToolSpec(
        server_name="chrome_devtools", tool_name="snapshot",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "chrome_devtools:list_console_messages": MCPToolSpec(
        server_name="chrome_devtools", tool_name="list_console_messages",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=5.0,
    ),
    "chrome_devtools:list_network_requests": MCPToolSpec(
        server_name="chrome_devtools", tool_name="list_network_requests",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=5.0,
    ),
    "chrome_devtools:wait_for": MCPToolSpec(
        server_name="chrome_devtools", tool_name="wait_for",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=30.0,
    ),

    # ── Chrome DevTools MCP — medium-risk interactions ────────────────────
    "chrome_devtools:evaluate_script": MCPToolSpec(
        server_name="chrome_devtools", tool_name="evaluate_script",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "chrome_devtools:click": MCPToolSpec(
        server_name="chrome_devtools", tool_name="click",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),

    # ── Chrome DevTools MCP — sensitive / always approval-gated ──────────
    "chrome_devtools:fill": MCPToolSpec(
        server_name="chrome_devtools", tool_name="fill",
        risk_level=RiskLevel.high, requires_approval=True,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "chrome_devtools:submit_form": MCPToolSpec(
        server_name="chrome_devtools", tool_name="submit_form",
        risk_level=RiskLevel.critical, requires_approval=True,
        max_calls_per_session=2, timeout_seconds=15.0,
    ),
    "chrome_devtools:upload_file": MCPToolSpec(
        server_name="chrome_devtools", tool_name="upload_file",
        risk_level=RiskLevel.critical, requires_approval=True,
        max_calls_per_session=2, timeout_seconds=30.0,
    ),

    # ── Playwright MCP — observation / read-only ──────────────────────────
    "playwright_mcp:browser_navigate": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_navigate",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=30, timeout_seconds=15.0,
    ),
    "playwright_mcp:browser_snapshot": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_snapshot",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_screenshot": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_screenshot",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_get_console": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_get_console",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=5.0,
    ),
    "playwright_mcp:browser_network_requests": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_network_requests",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=5.0,
    ),
    "playwright_mcp:browser_wait_for": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_wait_for",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=30.0,
    ),
    "playwright_mcp:browser_go_back": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_go_back",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_go_forward": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_go_forward",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_reload": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_reload",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_new_tab": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_new_tab",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_close_tab": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_close_tab",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=5, timeout_seconds=5.0,
    ),
    "playwright_mcp:browser_resize": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_resize",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=5, timeout_seconds=5.0,
    ),

    # ── Playwright MCP — medium-risk interactions ─────────────────────────
    "playwright_mcp:browser_click": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_click",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_hover": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_hover",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_drag": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_drag",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_press_key": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_press_key",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_evaluate": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_evaluate",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_select_option": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_select_option",
        risk_level=RiskLevel.medium, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),

    # ── Playwright MCP — sensitive / approval-gated ───────────────────────
    "playwright_mcp:browser_type": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_type",
        risk_level=RiskLevel.high, requires_approval=True,
        max_calls_per_session=10, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_fill": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_fill",
        risk_level=RiskLevel.high, requires_approval=True,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_check": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_check",
        risk_level=RiskLevel.high, requires_approval=True,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_uncheck": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_uncheck",
        risk_level=RiskLevel.high, requires_approval=True,
        max_calls_per_session=5, timeout_seconds=10.0,
    ),
    "playwright_mcp:browser_upload_file": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_upload_file",
        risk_level=RiskLevel.critical, requires_approval=True,
        max_calls_per_session=2, timeout_seconds=30.0,
    ),
    "playwright_mcp:browser_close": MCPToolSpec(
        server_name="playwright_mcp", tool_name="browser_close",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=5, timeout_seconds=5.0,
    ),

    # ── Scrapling MCP — web fetch / search ───────────────────────────────
    "scrapling_mcp:scrapling_fetch": MCPToolSpec(
        server_name="scrapling_mcp", tool_name="scrapling_fetch",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=30.0,
    ),
    "scrapling_mcp:scrapling_fetch_stealth": MCPToolSpec(
        server_name="scrapling_mcp", tool_name="scrapling_fetch_stealth",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=60.0,
    ),
    "scrapling_mcp:scrapling_search": MCPToolSpec(
        server_name="scrapling_mcp", tool_name="scrapling_search",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=10, timeout_seconds=30.0,
    ),

    # ── OpenRouter MCP ────────────────────────────────────────────────────
    "openrouter_mcp:ask_model": MCPToolSpec(
        server_name="openrouter_mcp", tool_name="ask_model",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=90.0,
    ),

    # ── Groq MCP (via llm_mcp_server.py) ─────────────────────────────────
    "groq_mcp:complete": MCPToolSpec(
        server_name="groq_mcp", tool_name="complete",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=65.0,
    ),

    # ── DeepInfra MCP (via llm_mcp_server.py) ────────────────────────────
    "deepinfra_mcp:complete": MCPToolSpec(
        server_name="deepinfra_mcp", tool_name="complete",
        risk_level=RiskLevel.low, requires_approval=False,
        max_calls_per_session=20, timeout_seconds=65.0,
    ),
}

# ── Subtask allowlists ────────────────────────────────────────────────────────
# Every broker call must name a subtask; only the listed tools are permitted.

SUBTASK_ALLOWLISTS: dict[str, SubtaskAllowlist] = {
    # Preview QA: open generated HTML in Chrome, screenshot, inspect errors
    "website_preview_qa": SubtaskAllowlist(
        subtask="website_preview_qa",
        allowed_server="chrome_devtools",
        allowed_tools=[
            "new_page",
            "navigate_page",
            "take_screenshot",
            "evaluate_script",
            "list_console_messages",
            "list_network_requests",
        ],
        requires_approval_for_any=False,
        max_calls_per_session=20,
    ),

    # DOM observation for browser-agent steps (read DOM / a11y tree)
    "agent_observe_dom": SubtaskAllowlist(
        subtask="agent_observe_dom",
        allowed_server="chrome_devtools",
        allowed_tools=["snapshot", "evaluate_script"],
        requires_approval_for_any=False,
        max_calls_per_session=20,
    ),

    # Screenshot-only observation (expensive — use sparingly)
    "agent_observe_screenshot": SubtaskAllowlist(
        subtask="agent_observe_screenshot",
        allowed_server="chrome_devtools",
        allowed_tools=["take_screenshot"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),

    # Navigate to a URL or wait for page state
    "agent_safe_navigation": SubtaskAllowlist(
        subtask="agent_safe_navigation",
        allowed_server="chrome_devtools",
        allowed_tools=["new_page", "navigate_page", "wait_for"],
        requires_approval_for_any=False,
        max_calls_per_session=30,
    ),

    # Click links/buttons (medium risk — no form submission)
    "agent_safe_click": SubtaskAllowlist(
        subtask="agent_safe_click",
        allowed_server="chrome_devtools",
        allowed_tools=["click", "wait_for"],
        requires_approval_for_any=False,
        max_calls_per_session=20,
    ),

    # Sensitive actions: no tool auto-approved — every call needs approval
    "sensitive_actions": SubtaskAllowlist(
        subtask="sensitive_actions",
        allowed_server="chrome_devtools",
        allowed_tools=[],   # empty: nothing may run without explicit approval
        requires_approval_for_any=True,
        max_calls_per_session=5,
    ),
    "agent_sensitive_action": SubtaskAllowlist(
        subtask="agent_sensitive_action",
        allowed_server="chrome_devtools",
        allowed_tools=["fill", "upload_file", "submit_form"],
        requires_approval_for_any=True,
        max_calls_per_session=5,
    ),

    # ── Playwright MCP subtasks ───────────────────────────────────────────
    "playwright_observe": SubtaskAllowlist(
        subtask="playwright_observe",
        allowed_server="playwright_mcp",
        allowed_tools=[
            "browser_snapshot", "browser_screenshot",
            "browser_get_console", "browser_network_requests",
        ],
        requires_approval_for_any=False,
        max_calls_per_session=20,
    ),
    "playwright_navigate": SubtaskAllowlist(
        subtask="playwright_navigate",
        allowed_server="playwright_mcp",
        allowed_tools=[
            "browser_navigate", "browser_wait_for",
            "browser_go_back", "browser_go_forward", "browser_reload",
            "browser_new_tab", "browser_close_tab", "browser_resize",
            "browser_close",
        ],
        requires_approval_for_any=False,
        max_calls_per_session=30,
    ),
    "playwright_interact": SubtaskAllowlist(
        subtask="playwright_interact",
        allowed_server="playwright_mcp",
        allowed_tools=[
            "browser_click", "browser_hover", "browser_drag",
            "browser_press_key", "browser_evaluate", "browser_select_option",
            "browser_wait_for",
        ],
        requires_approval_for_any=False,
        max_calls_per_session=20,
    ),
    "playwright_sensitive_action": SubtaskAllowlist(
        subtask="playwright_sensitive_action",
        allowed_server="playwright_mcp",
        allowed_tools=["browser_type", "browser_fill", "browser_check",
                       "browser_uncheck", "browser_upload_file"],
        requires_approval_for_any=True,
        max_calls_per_session=5,
    ),

    # ── Scrapling MCP subtasks ────────────────────────────────────────────
    "web_fetch": SubtaskAllowlist(
        subtask="web_fetch",
        allowed_server="scrapling_mcp",
        allowed_tools=["scrapling_fetch", "scrapling_fetch_stealth"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),
    "web_search": SubtaskAllowlist(
        subtask="web_search",
        allowed_server="scrapling_mcp",
        allowed_tools=["scrapling_search"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),

    # ── LLM delegation subtasks ───────────────────────────────────────────
    "llm_delegate_openrouter": SubtaskAllowlist(
        subtask="llm_delegate_openrouter",
        allowed_server="openrouter_mcp",
        allowed_tools=["ask_model"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),
    "llm_delegate_groq": SubtaskAllowlist(
        subtask="llm_delegate_groq",
        allowed_server="groq_mcp",
        allowed_tools=["complete"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),
    "llm_delegate_deepinfra": SubtaskAllowlist(
        subtask="llm_delegate_deepinfra",
        allowed_server="deepinfra_mcp",
        allowed_tools=["complete"],
        requires_approval_for_any=False,
        max_calls_per_session=10,
    ),
}


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_allowlist(subtask: str) -> SubtaskAllowlist | None:
    """Return the SubtaskAllowlist for the given subtask name, or None."""
    return SUBTASK_ALLOWLISTS.get(subtask)


def get_tool_spec(server_name: str, tool_name: str) -> MCPToolSpec | None:
    """Return the MCPToolSpec for a server+tool pair, or None."""
    return TOOL_REGISTRY.get(f"{server_name}:{tool_name}")
