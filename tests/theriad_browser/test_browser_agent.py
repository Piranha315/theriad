from theriad_browser.browser_agent import BrowserAgentStartRequest, STORE


def advance(session, ticks=10):
    for _ in range(ticks):
        session._last_tick = 0
        session.tick()


def test_mock_agent_requests_permission_and_approval_completes():
    session = STORE.start(BrowserAgentStartRequest(goal="submit contact form"))
    advance(session)
    assert session.status == "waiting_for_approval"
    assert session.pending_approval.proposed_action == "submit_form"
    session.approve()
    assert session.status == "completed"
    assert session.final_result


def test_deny_does_not_perform_sensitive_action():
    session = STORE.start(BrowserAgentStartRequest(goal="submit contact form"))
    advance(session)
    session.deny()
    assert session.status == "stopped"
    assert session.pending_approval is None
    assert "did not perform" in session.final_result
