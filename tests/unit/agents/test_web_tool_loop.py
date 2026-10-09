"""Model-driven web access: tool loop contract, provider payloads, gates."""
from __future__ import annotations

import json

import httpx

from theriad.agents import chat_web_access
from theriad.agents.web_tool_loop import (
    SandboxBrowserTools,
    WebLoopBudget,
    format_sources_footer,
    run_web_tool_loop,
)
from theriad.core.providers.interfaces import CompletionRequest, Message
from theriad.core.providers.openrouter import OpenRouterProvider


def _choice(content="", tool_calls=None, annotations=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if annotations:
        message["annotations"] = annotations
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def _call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _provider(responses, sent):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = queue.pop(0)
        body.setdefault("model", "test/model")
        body.setdefault("usage", {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001})
        return httpx.Response(200, json=body)

    return OpenRouterProvider(api_key="test", transport=httpx.MockTransport(handler))


def _request():
    return CompletionRequest(
        model="test/model",
        messages=[Message(role="system", content="base"), Message(role="user", content="weather in Delhi?")],
        max_tokens=256,
    )


async def test_server_tools_are_sent_and_citations_become_sources():
    sent: list[dict] = []
    answer = _choice(
        "It is 31°C in Delhi.",
        annotations=[{"type": "url_citation", "url_citation": {"url": "https://weather.example/delhi", "title": "Delhi weather"}}],
    )
    answer["usage"] = {
        "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.008,
        "server_tool_use": {"web_search_requests": 1},
    }
    provider = _provider([answer], sent)

    result = await run_web_tool_loop(provider, _request(), usage_sink=[])

    tools = sent[0]["tools"]
    assert {"type": "openrouter:web_search", "parameters": {"max_results": 5, "max_uses": 4}} in tools
    assert {"type": "openrouter:datetime"} in tools
    assert any(t.get("function", {}).get("name") == "web_fetch" for t in tools)
    assert not any(t.get("function", {}).get("name", "").startswith("browser_") for t in tools)
    assert sent[0]["tool_choice"] == "auto"
    assert "internet access" in sent[0]["messages"][0]["content"]
    assert result.sources == [{"url": "https://weather.example/delhi", "title": "Delhi weather"}]
    assert result.steps == [{"type": "search", "count": 1}]
    assert result.response.content == "It is 31°C in Delhi."


async def test_function_tool_round_trip_carries_call_ids():
    sent: list[dict] = []
    provider = _provider(
        [_choice(tool_calls=[_call("web_fetch", {"url": "https://example.com"})]), _choice("Example Domain is a test page.")],
        sent,
    )
    fetched: list[str] = []

    async def fake_fetch(url):
        fetched.append(url)
        return {"url": url, "status": "fetched", "text": "Example Domain"}

    result = await run_web_tool_loop(provider, _request(), fetch=fake_fetch, usage_sink=[])

    assert fetched == ["https://example.com"]
    second = sent[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1] == {
        "role": "tool",
        "content": json.dumps({"url": "https://example.com", "status": "fetched", "text": "Example Domain"}),
        "tool_call_id": "call_1",
    }
    assert result.sources == [{"url": "https://example.com", "title": "https://example.com"}]
    assert result.response.prompt_tokens == 20 and result.response.cost_usd == 0.002


async def test_budget_forces_final_answer_without_tools():
    sent: list[dict] = []
    fetch_call = _choice(tool_calls=[_call("web_fetch", {"url": "https://a.example"})])
    provider = _provider([fetch_call, fetch_call, _choice("Best answer from what I read.")], sent)

    async def fake_fetch(url):
        return {"url": url, "status": "fetched", "text": "x"}

    result = await run_web_tool_loop(
        provider, _request(), budget=WebLoopBudget(max_turns=3), fetch=fake_fetch, usage_sink=[]
    )

    assert [body["tool_choice"] for body in sent] == ["auto", "auto", "none"]
    assert result.response.content == "Best answer from what I read."


async def test_web_fetch_refuses_private_and_metadata_addresses():
    from theriad.agents.web_tool_loop import _web_fetch

    result = await _web_fetch("http://169.254.169.254/latest/meta-data/")
    assert result["status"] == "blocked_url"
    assert result["text"] == ""


class _FakeRuntime:
    def __init__(self):
        self.denied = False
        self.stopped = False
        self.sessions = {}

    async def start_session(self, **kwargs):
        return {"ok": True, "session_id": "s1"}

    async def observe(self, session_id, **kwargs):
        return {"status": "running", "current_url": "https://shop.example", "visible_text_summary": "Shop", "elements": [{"id": "osb-1", "text": "Buy now"}]}

    async def action(self, session_id, action=None, *, deny=False, **kwargs):
        if deny:
            self.denied = True
            return {"status": "denied"}
        return {"status": "waiting_for_approval", "pending_action": {"action": "purchase", "reason": "Purchases need approval."}}

    async def stop_session(self, session_id, **kwargs):
        self.stopped = True


async def test_side_effecting_browser_actions_are_denied_and_session_closed():
    runtime = _FakeRuntime()
    browser = SandboxBrowserTools(runtime, goal="g", owner_user_id="u1", owner_organization_id="o1")
    sent: list[dict] = []
    provider = _provider(
        [
            _choice(tool_calls=[_call("browser_open", {"url": "https://shop.example"}, "c1")]),
            _choice(tool_calls=[_call("browser_click", {"element_id": "osb-1"}, "c2")]),
            _choice("I can't buy things from chat."),
        ],
        sent,
    )

    result = await run_web_tool_loop(provider, _request(), browser=browser, usage_sink=[])

    assert any(t.get("function", {}).get("name") == "browser_open" for t in sent[0]["tools"])
    click_result = json.loads(sent[2]["messages"][-1]["content"])
    assert click_result["status"] == "denied"
    assert runtime.denied and runtime.stopped
    assert [step["tool"] for step in result.steps] == ["browser_open", "browser_click"]


async def test_browser_open_rejects_private_hosts_without_starting_sandbox():
    runtime = _FakeRuntime()
    browser = SandboxBrowserTools(runtime, goal="g", owner_user_id=None, owner_organization_id=None)
    result = await browser.call("browser_open", {"url": "http://127.0.0.1:8080/health"})
    assert result["status"] == "blocked_url"
    assert browser.session_id is None


def test_web_tools_gate(monkeypatch):
    monkeypatch.delenv("THERIAD_CHAT_WEB_TOOLS", raising=False)
    assert chat_web_access.web_tools_enabled(guest=False, execution_provider="theriad")
    assert not chat_web_access.web_tools_enabled(guest=True, execution_provider="theriad")
    assert not chat_web_access.web_tools_enabled(guest=False, execution_provider="anthropic")
    monkeypatch.setenv("THERIAD_CHAT_WEB_TOOLS", "0")
    assert not chat_web_access.web_tools_enabled(guest=False, execution_provider="theriad")


async def test_hosted_browser_requires_flag_and_health(monkeypatch):
    chat_web_access.reset_for_tests()
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.delenv("THERIAD_HOSTED_WEB_BROWSING", raising=False)

    class Healthy(_FakeRuntime):
        async def availability(self, probe=True):
            return {"available": True}

    assert await chat_web_access.browser_tools_for_turn(
        goal="g", owner_user_id="u", owner_organization_id="o", runtime=Healthy()
    ) is None
    monkeypatch.setenv("THERIAD_HOSTED_WEB_BROWSING", "1")
    tools = await chat_web_access.browser_tools_for_turn(
        goal="g", owner_user_id="u", owner_organization_id="o", runtime=Healthy()
    )
    assert isinstance(tools, SandboxBrowserTools)

    chat_web_access.reset_for_tests()

    class Down(_FakeRuntime):
        async def availability(self, probe=True):
            return {"available": False}

    assert await chat_web_access.browser_tools_for_turn(
        goal="g", owner_user_id="u", owner_organization_id="o", runtime=Down()
    ) is None
    chat_web_access.reset_for_tests()


def test_sources_footer_skips_already_linked_urls():
    sources = [{"url": "https://a.example", "title": "A"}, {"url": "https://b.example", "title": "B"}]
    footer = format_sources_footer(sources, "See [A](https://a.example).")
    assert "https://b.example" in footer and "https://a.example" not in footer
    assert format_sources_footer(sources[:1], "See https://a.example") == ""


async def test_search_count_read_from_server_tool_use_details_and_markup_stripped():
    sent: list[dict] = []
    answer = _choice(
        'NVDA closed at $225.07【commentary to=functions.openrouter_web_search <|constrain|>json<|message|>{"query":"x"}'
    )
    answer["usage"] = {
        "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.007,
        "server_tool_use_details": {"web_search_requests": 2},
    }
    result = await run_web_tool_loop(_provider([answer], sent), _request(), usage_sink=[])
    assert result.steps == [{"type": "search", "count": 2}]
    assert result.response.content == "NVDA closed at $225.07"
