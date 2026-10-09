"""
Unit tests for theriad.tools.mcp_broker.MCPBroker

All tests use stub adapters — no real Chrome MCP process or network.

Run with:
    pytest tests/unit/tools/test_mcp_broker.py -v
"""
from __future__ import annotations

import pytest

from theriad.tools.mcp_broker import MCPBroker
from theriad.tools.mcp_schemas import CallStatus


# ── Stub adapters ─────────────────────────────────────────────────────────────

class StubAdapter:
    """Available adapter that echoes tool name + arguments as result."""
    available = True
    connected = True
    error     = None
    tools     = ["take_screenshot", "navigate_page", "new_page",
                 "list_console_messages", "list_network_requests",
                 "evaluate_script", "snapshot", "click", "wait_for"]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return {"tool": tool_name, "args": arguments, "status": "ok"}


class FailingAdapter:
    """Available adapter whose call_tool always raises."""
    available = True
    connected = True
    error     = None
    tools     = ["take_screenshot"]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        raise RuntimeError("simulated adapter failure")


class UnavailableAdapter:
    """Adapter that reports itself unavailable (MCP not installed)."""
    available = False
    connected = False
    error     = "Chrome MCP not configured."
    tools: list = []

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        raise RuntimeError("should never be called")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def broker() -> MCPBroker:
    return MCPBroker(
        session_id="test-session",
        adapters={"chrome_devtools": StubAdapter()},
    )


# ── 1. Happy-path allowed call ────────────────────────────────────────────────

def test_allowed_tool_succeeds(broker):
    """A tool in the subtask allowlist executes and returns ok=True."""
    result = broker.call(
        subtask="website_preview_qa",
        server_name="chrome_devtools",
        tool_name="take_screenshot",
        arguments={"page_id": "p1"},
    )
    assert result.ok is True
    assert result.result is not None
    assert result.result["tool"] == "take_screenshot"
    assert result.error is None
    assert result.blocked_reason is None


def test_allowed_navigation_tool(broker):
    """navigate_page is allowed for agent_safe_navigation."""
    result = broker.call(
        subtask="agent_safe_navigation",
        server_name="chrome_devtools",
        tool_name="navigate_page",
        arguments={"url": "https://example.com"},
    )
    assert result.ok is True


# ── 2. Allowlist blocking ─────────────────────────────────────────────────────

def test_tool_not_in_allowlist_is_blocked(broker):
    """A tool valid on the server but not in the subtask allowlist is blocked."""
    # "click" is NOT in website_preview_qa allowlist
    result = broker.call(
        subtask="website_preview_qa",
        server_name="chrome_devtools",
        tool_name="click",
    )
    assert result.ok is False
    assert result.blocked_reason is not None
    assert "allowlist" in result.blocked_reason.lower()


def test_wrong_server_is_blocked(broker):
    """A call to a server that doesn't match the subtask is blocked."""
    result = broker.call(
        subtask="website_preview_qa",
        server_name="some_other_server",
        tool_name="take_screenshot",
    )
    assert result.ok is False
    assert result.blocked_reason is not None


def test_unknown_subtask_is_blocked(broker):
    """An unregistered subtask name is rejected immediately."""
    result = broker.call(
        subtask="nonexistent_subtask",
        server_name="chrome_devtools",
        tool_name="take_screenshot",
    )
    assert result.ok is False
    assert result.blocked_reason is not None


# ── 3. Approval gate ──────────────────────────────────────────────────────────

def test_sensitive_subtask_requires_approval(broker):
    """The 'sensitive_actions' subtask never auto-approves — always approval_pending."""
    result = broker.call(
        subtask="sensitive_actions",
        server_name="chrome_devtools",
        tool_name="take_screenshot",   # even a read-only tool is blocked here
    )
    assert result.ok is False
    # tool not in allowlist → blocked_reason (not approval_required)
    # because the allowlist is empty for sensitive_actions
    assert result.blocked_reason is not None


def test_high_risk_tool_requires_approval():
    """A high-risk tool (fill) marked requires_approval=True triggers approval gate."""
    # We need to inject fill into an allowlist for this test. We test the
    # approval gate via the registry spec's requires_approval flag.
    from theriad.tools.mcp_registry import SUBTASK_ALLOWLISTS, TOOL_REGISTRY, SubtaskAllowlist
    import copy

    # Build a custom subtask that permits "fill"
    test_subtask = SubtaskAllowlist(
        subtask="_test_fill",
        allowed_server="chrome_devtools",
        allowed_tools=["fill"],
        requires_approval_for_any=False,
        max_calls_per_session=5,
    )
    original = SUBTASK_ALLOWLISTS.copy()
    try:
        SUBTASK_ALLOWLISTS["_test_fill"] = test_subtask
        b = MCPBroker("t1", {"chrome_devtools": StubAdapter()})
        result = b.call("_test_fill", "chrome_devtools", "fill", {"selector": "#pw", "value": "s3cr3t"})
        assert result.ok is False
        assert result.approval_required is True
    finally:
        SUBTASK_ALLOWLISTS.clear()
        SUBTASK_ALLOWLISTS.update(original)


# ── 4. Budget enforcement ─────────────────────────────────────────────────────

def test_budget_exceeded_blocks_further_calls():
    """After the per-session budget is exhausted, calls are blocked."""
    # agent_observe_screenshot allows take_screenshot with max 10/session
    b = MCPBroker("budget-test", {"chrome_devtools": StubAdapter()})
    for _ in range(10):
        r = b.call("agent_observe_screenshot", "chrome_devtools", "take_screenshot")
        assert r.ok is True
    # 11th call must be blocked
    r11 = b.call("agent_observe_screenshot", "chrome_devtools", "take_screenshot")
    assert r11.ok is False
    assert "budget" in (r11.blocked_reason or "").lower()


# ── 5. Adapter errors ────────────────────────────────────────────────────────

def test_adapter_exception_returns_error_result():
    """When the adapter raises, the broker returns ok=False with the error text."""
    b = MCPBroker("err-test", {"chrome_devtools": FailingAdapter()})
    result = b.call("agent_observe_screenshot", "chrome_devtools", "take_screenshot")
    assert result.ok is False
    assert result.error is not None
    assert "simulated adapter failure" in result.error


def test_unavailable_adapter_blocks_call():
    """An unavailable adapter blocks the call without raising."""
    b = MCPBroker("unav-test", {"chrome_devtools": UnavailableAdapter()})
    result = b.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    assert result.ok is False
    assert result.blocked_reason is not None


def test_missing_adapter_blocks_call(broker):
    """A call to a server with no registered adapter is blocked gracefully."""
    b = MCPBroker("no-adapter", {})   # no adapters at all
    result = b.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    assert result.ok is False
    assert result.blocked_reason is not None


# ── 6. Call log ───────────────────────────────────────────────────────────────

def test_call_log_records_every_attempt(broker):
    """Every call attempt — success or failure — is appended to the call log."""
    # Success
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    # Blocked
    broker.call("website_preview_qa", "chrome_devtools", "click")
    # Unknown subtask
    broker.call("bogus", "chrome_devtools", "take_screenshot")

    log = broker.get_call_log()
    assert len(log) == 3


def test_call_log_statuses(broker):
    """Log entries carry the correct status for success and blocked calls."""
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    broker.call("website_preview_qa", "chrome_devtools", "click")   # blocked

    log = broker.get_call_log()
    statuses = {e.status for e in log}
    assert CallStatus.ok in statuses
    assert CallStatus.blocked in statuses


def test_call_log_is_copy(broker):
    """get_call_log() returns a copy — mutations don't affect the broker's log."""
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    copy = broker.get_call_log()
    copy.clear()
    assert len(broker.get_call_log()) == 1


# ── 7. Broker status ─────────────────────────────────────────────────────────

def test_get_status_lists_known_subtasks(broker):
    """get_status() includes all registered subtask names."""
    from theriad.tools.mcp_registry import SUBTASK_ALLOWLISTS
    status = broker.get_status()
    assert set(status.subtask_allowlists) == set(SUBTASK_ALLOWLISTS.keys())


def test_get_status_available_adapter(broker):
    """A live stub adapter shows available=True in broker status."""
    status = broker.get_status()
    assert len(status.servers) == 1
    assert status.servers[0].available is True
    assert status.servers[0].server_name == "chrome_devtools"


def test_get_status_unavailable_adapter():
    """An unavailable adapter shows available=False with an error message."""
    b = MCPBroker("s1", {"chrome_devtools": UnavailableAdapter()})
    status = b.get_status()
    assert status.servers[0].available is False
    assert status.servers[0].error is not None


def test_get_status_call_counter(broker):
    """total_calls_this_session counts only successful executions."""
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    broker.call("website_preview_qa", "chrome_devtools", "click")    # blocked, no increment
    status = broker.get_status()
    assert status.total_calls_this_session == 2


# ── 8. Chrome DevTools adapter availability ──────────────────────────────────

def test_chrome_adapter_unavailable_without_config(monkeypatch):
    """Without CHROME_MCP_ENABLED or CHROME_MCP_WS_URL, adapter is unavailable."""
    # Clear both the current (THERIAD_*) and legacy env var names — server.py
    # calls load_dotenv() on import, which can populate THERIAD_CHROME_MCP_ENABLED
    # from the repo's local .env into the real process environment (not
    # monkeypatch-scoped) if a test that imports theriad.api.server ran earlier
    # in the same pytest process. Clearing all three keeps this test
    # deterministic regardless of import order.
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL",  raising=False)

    from theriad.tools import chrome_devtools_adapter as mod
    # Re-instantiate to pick up env change
    adapter = mod.ChromeDevToolsAdapter()
    assert adapter.available is False
    assert adapter.error is not None


def test_chrome_adapter_available_with_env_flag(monkeypatch):
    """CHROME_MCP_ENABLED=true marks the adapter as available."""
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.setenv("CHROME_MCP_ENABLED", "true")
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)

    from theriad.tools import chrome_devtools_adapter as mod
    adapter = mod.ChromeDevToolsAdapter()
    assert adapter.available is True
    assert adapter.error is None


def test_chrome_adapter_available_with_ws_url(monkeypatch):
    """CHROME_MCP_WS_URL takes precedence over CHROME_MCP_ENABLED."""
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.setenv("CHROME_MCP_WS_URL", "ws://localhost:9222/json")

    from theriad.tools import chrome_devtools_adapter as mod
    adapter = mod.ChromeDevToolsAdapter()
    assert adapter.available is True


def test_chrome_adapter_call_tool_raises_when_transport_fails(monkeypatch):
    """
    call_tool raises RuntimeError when the MCP subprocess cannot be started
    or the initialize handshake times out.

    The transport is now wired (MCPStdioClient over subprocess stdio).
    We patch MCPStdioClient.__init__ to raise immediately so the test is fast
    and offline, while still exercising the error-propagation path.
    """
    monkeypatch.setenv("CHROME_MCP_ENABLED", "true")

    # Patch MCPStdioClient to simulate a subprocess start failure
    from theriad.tools import mcp_stdio_client as stdio_mod, chrome_devtools_adapter as mod
    from theriad.tools.mcp_stdio_client import MCPError

    def _failing_init(self, command, **kwargs):
        raise MCPError(-32000, "Subprocess not available in test environment")

    monkeypatch.setattr(stdio_mod.MCPStdioClient, "__init__", _failing_init)

    adapter = mod.ChromeDevToolsAdapter()
    with pytest.raises(RuntimeError):
        adapter.call_tool("take_screenshot", {})


def test_chrome_adapter_unavailable_call_raises_runtime(monkeypatch):
    """call_tool on an unavailable adapter raises RuntimeError, not NotImplementedError."""
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL",  raising=False)

    from theriad.tools import chrome_devtools_adapter as mod
    adapter = mod.ChromeDevToolsAdapter()
    with pytest.raises(RuntimeError):
        adapter.call_tool("take_screenshot", {})
