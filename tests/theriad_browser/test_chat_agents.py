import pytest
from fastapi.testclient import TestClient

from theriad_browser.browser_runtime import BrowserAgentSession
from theriad_browser.chat import app, browser_runtime, handle_chat_message
from theriad_browser.intent import ChatIntent, classify_chat_intent
from theriad_browser.tool_broker import ToolBroker, ToolPolicyError


def test_browser_like_chat_prompt_routes_to_browser_agent():
    assert classify_chat_intent("Go to example.com and summarize it") == ChatIntent.BROWSER_AGENT


def test_terminal_like_chat_prompt_routes_to_terminal_agent():
    assert classify_chat_intent("Run pytest and summarize failures") == ChatIntent.TERMINAL_AGENT


def test_normal_prompt_stays_normal_chat():
    assert classify_chat_intent("What is cloud cost optimization?") == ChatIntent.NORMAL_CHAT


def test_browser_agent_chat_run_creates_browser_agent_session():
    response = handle_chat_message("visit example.com")
    assert response.agent_run_id in browser_runtime.sessions
    assert isinstance(browser_runtime.sessions[response.agent_run_id], BrowserAgentSession)


def test_terminal_tool_does_not_execute_commands_in_mock_mode(tmp_path):
    broker = ToolBroker(tmp_path, terminal_mode="mock")
    result = broker.call("terminal.run_mock", command="git status")
    assert result.status == "mocked"
    assert broker.log[0]["command"] == "git status"


def test_safe_terminal_command_allowed_only_inside_project_root(tmp_path):
    broker = ToolBroker(tmp_path)
    assert broker.classify_command("pwd") == "safe"
    with pytest.raises(ToolPolicyError):
        broker.classify_command("pwd", cwd="../outside")


def test_dangerous_commands_are_blocked(tmp_path):
    broker = ToolBroker(tmp_path)
    with pytest.raises(ToolPolicyError):
        broker.terminal_run_mock("sudo rm -rf /")


def test_write_commands_require_approval(tmp_path):
    broker = ToolBroker(tmp_path)
    result = broker.terminal_run_mock("git commit -m test")
    assert result.requires_approval is True


def test_chat_ui_renders_agent_status_component_exists():
    source = open("frontend/src/ChatAgentStatus.jsx", encoding="utf-8").read()
    assert "data-testid=\"chat-agent-status\"" in source
    assert "waiting_for_approval" in source


def test_existing_agent_page_still_works():
    source = open("frontend/src/AgentDebugPanel.jsx", encoding="utf-8").read()
    assert "Agent Debug Trace" in source
    client = TestClient(app)
    created = client.post("/api/chat/send", json={"message": "browse example.com"}).json()
    trace = client.get(f"/api/agent/{created['agent_run_id']}/trace").json()
    assert trace["trace"]
