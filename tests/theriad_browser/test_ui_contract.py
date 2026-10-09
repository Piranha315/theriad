from pathlib import Path

FIXTURE_HTML = Path("static/index.html").read_text()
EXECUTE_RESULT = Path("frontend/src/ExecuteResult.jsx").read_text()
APP = Path("frontend/src/main.jsx").read_text()
AGENT_PANEL = Path("frontend/src/BrowserAgentPanel.jsx").read_text()


def test_react_root_has_the_canonical_agent_surface():
    assert '<BrowserAgentPanel />' in APP
    assert 'data-testid="browser-agent-panel"' in AGENT_PANEL
    assert "demo" not in APP


def test_agent_ui_has_compact_live_card_and_approval_sheet():
    assert "Current URL" in AGENT_PANEL
    assert "Page summary" in AGENT_PANEL
    assert 'data-testid="approval-sheet"' in AGENT_PANEL
    assert "Developer diagnostics" in AGENT_PANEL


def test_agent_ui_uses_adaptive_polling_and_caps_backend_activity():
    assert "POLL_DELAYS" in AGENT_PANEL
    assert "setTimeout" in AGENT_PANEL
    browser_agent = Path("theriad_browser/browser_agent.py").read_text()
    assert "del self.activity_log[:-50]" in browser_agent


def test_normal_ui_has_no_step_count_controls():
    assert "Run 1 step" not in AGENT_PANEL
    assert "Run 5 steps" not in AGENT_PANEL
    assert "Run until checkpoint" not in AGENT_PANEL


def test_static_page_is_explicitly_a_development_fixture():
    assert 'data-fixture="browser-agent"' in FIXTURE_HTML
    assert "not the Theriad product UI" in FIXTURE_HTML


def test_execute_result_renders_answer_separately_before_trace():
    answer_position = EXECUTE_RESULT.index('data-testid="execute-final-answer"')
    details_position = EXECUTE_RESULT.index("<summary>Execution details</summary>")

    assert "response.final_answer" in EXECUTE_RESULT
    assert answer_position < details_position
    assert "<AgentDebugPanel trace={response.trace || []} />" in EXECUTE_RESULT
    assert "Execution completed, but no final answer was generated." in EXECUTE_RESULT
