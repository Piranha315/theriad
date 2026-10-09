"""Regression coverage for the OpenSandbox-first Chrome MCP compatibility seam."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from theriad.browser.bridge import BrowserBridge
from theriad.tools.chrome_devtools_adapter import (
    ChromeDevToolsAdapter,
    _normalize_mcp_result,
)
from theriad.tools.chrome_mcp import ChromeMCP


class RecordingBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict, dict]] = []
        self.closed = False

    def call(self, subtask, server, tool, arguments, **kwargs):
        self.calls.append((subtask, server, tool, dict(arguments), dict(kwargs)))
        if tool == "new_page":
            result = {"pageId": 7}
        elif tool == "snapshot":
            result = {
                "url": "https://example.com",
                "title": "Example",
                "visible_text": "Example page",
                "links": [],
                "buttons": [],
            }
        elif tool == "take_screenshot":
            result = {"screenshot": "aGVsbG8="}
        elif tool == "evaluate_script":
            result = {"value": "https://example.com"}
        else:
            result = {"ok": True}
        return SimpleNamespace(
            ok=True,
            result=result,
            error=None,
            blocked_reason=None,
        )

    def close(self) -> None:
        self.closed = True


def test_facade_uses_operation_specific_broker_allowlists():
    broker = RecordingBroker()
    chrome = ChromeMCP(broker)

    chrome.new_page()
    chrome.navigate("https://example.com")
    chrome.read_dom()
    chrome.screenshot()
    chrome.click("uid=button-1")
    chrome.type("uid=input-1", "value", approved=True)

    assert [call[0] for call in broker.calls] == [
        "agent_safe_navigation",
        "agent_safe_navigation",
        "agent_observe_dom",
        "agent_observe_screenshot",
        "agent_safe_click",
        "agent_sensitive_action",
    ]
    assert broker.calls[-1][4] == {"approval_received": True}
    assert broker.calls[1][3]["pageId"] == 7
    chrome.close()
    assert broker.closed is True


def test_chrome_adapter_reuses_transport_and_attaches_ws(monkeypatch):
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.setenv(
        "CHROME_MCP_WS_URL",
        "ws://127.0.0.1:9222/devtools/browser/test",
    )

    import theriad.tools.mcp_stdio_client as stdio

    clients = []

    class FakeClient:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        def call_tool(self, tool_name, arguments):
            return {"tool": tool_name, "arguments": arguments}

        def close(self):
            self.closed = True

    monkeypatch.setattr(stdio, "MCPStdioClient", FakeClient)
    adapter = ChromeDevToolsAdapter()

    first = adapter.call_tool("snapshot", {})
    second = adapter.call_tool("evaluate_script", {"expression": "document.title"})
    click = adapter.call_tool("click", {"pageId": "7", "selector": "button-1"})
    fill = adapter.call_tool("fill", {"pageId": "7", "selector": "input-1", "value": "x"})

    assert len(clients) == 1
    ws_args = [arg for arg in clients[0].command if arg.startswith("--wsEndpoint=")]
    assert ws_args == [
        "--wsEndpoint=ws://127.0.0.1:9222/devtools/browser/test"
    ]
    assert first["tool"] == "take_snapshot"
    assert second["tool"] == "evaluate_script"
    assert second["arguments"]["function"].startswith("() =>")
    assert click["arguments"] == {"pageId": 7, "uid": "button-1"}
    assert fill["arguments"] == {"pageId": 7, "uid": "input-1", "value": "x"}

    adapter.close()
    assert clients[0].closed is True


def test_raw_mcp_content_is_flattened_without_losing_page_metadata():
    payload = {
        "pages": [{"pageId": 13, "url": "https://example.com"}],
        "content": [{"type": "text", "text": '{"title":"Example"}'}],
    }

    normalized = _normalize_mcp_result(payload)

    assert normalized["title"] == "Example"
    assert normalized["pages"][0]["pageId"] == 13
    assert normalized["pageId"] == 13
    assert normalized["value"] == {"title": "Example"}


def test_facade_reads_page_id_from_current_new_page_page_list():
    class ContentBroker:
        def __init__(self):
            self.calls = []

        def call(self, subtask, server, tool, arguments, **kwargs):
            self.calls.append((tool, dict(arguments)))
            if tool == "new_page":
                result = {
                    "pages": [{"pageId": 13, "url": "https://example.com"}],
                    "content": [{"type": "text", "text": '{"title":"Example"}'}],
                }
            else:
                result = {"ok": True}
            return SimpleNamespace(ok=True, result=result, error=None, blocked_reason=None)

    broker = ContentBroker()
    chrome = ChromeMCP(broker)

    page = chrome.new_page("https://example.com")
    chrome.navigate("https://example.com/next")

    assert page["pages"][0]["pageId"] == 13
    assert broker.calls[1][1]["pageId"] == 13


def test_facade_does_not_send_none_as_page_id_when_response_has_no_metadata():
    class MetadataFreeBroker:
        def __init__(self):
            self.calls = []

        def call(self, subtask, server, tool, arguments, **kwargs):
            self.calls.append((tool, dict(arguments)))
            return SimpleNamespace(
                ok=True,
                result={"content": [{"type": "text", "text": "Page opened."}]},
                error=None,
                blocked_reason=None,
            )

    broker = MetadataFreeBroker()
    chrome = ChromeMCP(broker)

    chrome.new_page()
    chrome.navigate("https://example.com")

    assert "pageId" not in broker.calls[1][1]


def test_boolean_chrome_flag_starts_isolated_without_fake_ws(monkeypatch):
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.setenv("CHROME_MCP_ENABLED", "true")
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)

    adapter = ChromeDevToolsAdapter()
    command = adapter._build_command(adapter._runtime_config)

    assert "--isolated" in command
    assert not any("9222/json" in arg for arg in command)


def test_current_chrome_flag_false_overrides_legacy_true(monkeypatch):
    monkeypatch.setenv("THERIAD_CHROME_MCP_ENABLED", "false")
    monkeypatch.setenv("CHROME_MCP_ENABLED", "true")
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)

    adapter = ChromeDevToolsAdapter()

    assert adapter.available is False


def test_http_chrome_endpoint_uses_browser_url_without_launch_flags(monkeypatch):
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.setenv("CHROME_MCP_WS_URL", "http://127.0.0.1:9222")

    adapter = ChromeDevToolsAdapter()
    command = adapter._build_command(adapter._runtime_config)

    assert "--browserUrl=http://127.0.0.1:9222" in command
    assert "--isolated" not in command


class UnavailableRuntime:
    async def availability(self):
        return {"available": False, "status": "unavailable"}


class HealthyRuntime:
    async def availability(self):
        return {"available": True, "status": "available"}

    async def start_session(self, **kwargs):
        return {
            "ok": True,
            "status": "running",
            "backend": "opensandbox",
            "session_id": kwargs.get("session_id") or "osb-session",
            "current_url": kwargs.get("start_url") or "about:blank",
        }


@pytest.mark.asyncio
async def test_legacy_injected_chrome_fallback_is_usable_when_opensandbox_is_down():
    broker = RecordingBroker()
    bridge = BrowserBridge(
        chrome_mcp=ChromeMCP(broker),
        runtime=UnavailableRuntime(),
    )

    started = await bridge.start_session({
        "session_id": "legacy-session",
        "start_url": "https://example.com",
    })
    assert started.ok is True
    assert started.backend == "chrome_mcp"

    observed = await bridge.read_page("legacy-session")
    clicked = await bridge.click("legacy-session", "uid=button-1")
    pending = await bridge.fill("legacy-session", "uid=input-1", "secret")
    filled = await bridge.fill(
        "legacy-session",
        "uid=input-1",
        "secret",
        approved=True,
    )

    assert observed.ok is True
    assert clicked.ok is True
    assert pending.status == "approval_required"
    assert filled.ok is True


@pytest.mark.asyncio
async def test_healthy_opensandbox_wins_over_explicit_legacy_injection():
    broker = RecordingBroker()
    bridge = BrowserBridge(
        chrome_mcp=ChromeMCP(broker),
        runtime=HealthyRuntime(),
    )

    started = await bridge.start_session({"session_id": "osb-session"})

    assert started.ok is True
    assert started.backend == "opensandbox"
    assert broker.calls == []


@pytest.mark.asyncio
async def test_legacy_api_search_does_not_call_opensandbox_handle(tmp_path, monkeypatch):
    import theriad.api.server as server
    from theriad.agents.task_engine import TaskEngine
    from theriad.search.service import SearchResult

    engine = TaskEngine(event_dir=tmp_path / "events", allowed_workspace_root=tmp_path)
    task = engine.create(
        goal="research compatibility",
        workspace_root=str(tmp_path),
        metadata={
            "browser_backend": "chrome_mcp",
            "browser_session_id": "legacy-session",
        },
    )
    monkeypatch.setattr(server, "get_task_engine", lambda: engine)

    async def fake_search(_service, query):
        assert query == "research compatibility"
        return [SearchResult("Docs", "https://example.com/docs", "", "duckduckgo")]

    monkeypatch.setattr("theriad.search.service.SearchService.search", fake_search)
    cleanup_calls = []

    async def fake_cleanup(task_id):
        cleanup_calls.append(task_id)
        return {"ok": True, "status": "stopped", "backend": "chrome_mcp"}

    monkeypatch.setattr(engine, "cleanup_browser_session", fake_cleanup)

    result = await server._complete_browser_api_search(task)

    assert result["ok"] is True
    assert result["backend"] == "chrome_mcp"
    assert task.status == "completed"
    assert task.final_answer == "API search results: Docs: https://example.com/docs"
    assert cleanup_calls == [task.task_id]
