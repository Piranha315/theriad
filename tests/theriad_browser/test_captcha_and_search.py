from pathlib import Path
from asyncio import run

from theriad_browser.browser_agent import AgentStatus, BrowserAgentStartRequest, STORE
from theriad_browser.observation import BrowserObservation, is_captcha_observation
from theriad.execution import run_execute_flow
from theriad.provider import ProviderResult
from theriad.routing import resolve_route
from theriad.search import SearchResult


def test_captcha_page_fixture_triggers_needs_approval():
    html = Path("tests/fixtures/google_captcha.html").read_text()
    session = STORE.start(BrowserAgentStartRequest(goal="search Google"))

    detected = session.observe_page(
        BrowserObservation(
            current_url="https://www.google.com/search?q=theriad_browser",
            page_title="Before you continue to Google Search",
            visible_text=html,
            screenshot="data:image/png;base64,fixture",
        )
    )

    assert detected is True
    assert session.status == AgentStatus.NEEDS_APPROVAL
    assert session.current_url == "https://www.google.com/search?q=theriad_browser"
    assert session.page_title == "Before you continue to Google Search"
    assert session.screenshot
    assert session.final_result
    assert "CAPTCHA encountered" in session.final_result


def test_google_sorry_url_triggers_needs_approval_without_text():
    observation = BrowserObservation(
        current_url="https://www.google.com/sorry/index?continue=search",
        page_title="",
        visible_text="",
    )

    assert is_captcha_observation(observation) is True


def test_agent_does_not_retry_captcha_page():
    session = STORE.start(BrowserAgentStartRequest(goal="search Google"))
    session.observe_page(
        BrowserObservation(
            current_url="https://google.com/sorry/index",
            page_title="Google",
            visible_text="reCAPTCHA",
        )
    )
    diagnostics_count = len(session.diagnostics)

    for _ in range(5):
        session._last_tick = 0
        session.tick()

    assert session.status == AgentStatus.NEEDS_APPROVAL
    assert len(session.diagnostics) == diagnostics_count


def test_captcha_handoff_can_switch_to_api_search_without_retrying():
    session = STORE.start(BrowserAgentStartRequest(goal="search for Theriad"))
    session.observe_page(
        BrowserObservation(
            current_url="https://google.com/sorry/index",
            page_title="Google CAPTCHA",
            visible_text="unusual traffic from your computer network",
        )
    )

    session.use_api_search()

    assert session.status == AgentStatus.COMPLETED
    assert session.captcha_detected is False
    assert session.final_result
    assert session.artifacts


def test_web_search_prompt_calls_search_service_before_provider(monkeypatch):
    calls = []

    def mocked_search(self, query):
        calls.append("search")
        return [
            SearchResult(
                title="Theriad",
                url="https://example.com/theriad_browser",
                snippet="API-first result",
                source="mock-test",
            )
        ]

    async def mocked_provider(prompt, route, temperature=None, max_tokens=None):
        calls.append("provider")
        return ProviderResult(
            content="Theriad result from https://example.com/theriad_browser",
            prompt_tokens=1,
            completion_tokens=1,
            estimated_cost_usd=0,
            actual_cost_usd=0,
            latency_ms=1,
            fallback_used=False,
        )

    monkeypatch.setattr("theriad.execution.SearchService.search", mocked_search)
    monkeypatch.setattr("theriad.execution.execute_with_fallback", mocked_provider)

    evidence, final_answer = run(
        run_execute_flow(
            "Search the web for Theriad",
            resolve_route("balanced"),
        )
    )

    assert calls == ["search", "provider"]
    assert evidence.trace[0]["type"] == "search"
    assert evidence.artifacts[0]["url"] == "https://example.com/theriad_browser"
    assert final_answer


def test_captcha_frontend_has_all_handoff_actions():
    html = Path("static/index.html").read_text()

    assert "CAPTCHA encountered. Complete it manually or switch to API search." in html
    assert "I completed it, continue" in html
    assert "Use API search instead" in html
    assert "Stop task" in html
    assert "captchaScreenshot" in html
