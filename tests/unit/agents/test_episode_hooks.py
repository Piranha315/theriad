"""Safety and accounting invariants for actual opt-in episode entrypoints."""
from __future__ import annotations

import asyncio
import json

import pytest

from theriad.agents.model_telemetry import complete_with_telemetry
from theriad.core.providers.interfaces import CompletionRequest, CompletionResponse, Message
from theriad.routing.episode_hooks import (
    EpisodeRoutingError,
    EpisodeRoutingSession,
    JsonlAttemptSink,
    current_episode,
    episode_routing,
    recover_attempts,
)

MODEL = "fixture/challenger"
METADATA = {MODEL: {"capabilities": ["text"], "input_per_1m": 2.0, "output_per_1m": 4.0}}


class Provider:
    name = "openrouter"

    def __init__(self, outcomes=None):
        self.requests = []
        self.outcomes = list(outcomes or [])

    async def complete(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else "done"
        if isinstance(outcome, BaseException):
            raise outcome
        return CompletionResponse(
            model=request.model, content=outcome, prompt_tokens=100, completion_tokens=50,
            latency_ms=1.0, provider=self.name, finish_reason="stop",
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 50,
                           "completion_tokens_details": {"reasoning_tokens": 12}}},
        )


def request(model=MODEL, **kwargs):
    return CompletionRequest(model=model, messages=[Message("user", "private input")], **kwargs)


def test_resolver_runs_once_and_freezes_baseline_and_configuration():
    calls = []
    selection = {"planner": MODEL}

    def resolver(context):
        calls.append(context)
        with pytest.raises(TypeError):
            context.baseline_models["planner"] = "changed"
        return selection

    session = EpisodeRoutingSession("e", resolver=resolver, model_metadata=METADATA)
    session.prepare("agents", {"planner": "baseline"}, "theriad", "initial prompt")
    selection["planner"] = "changed"
    session.prepare("agents", {"planner": "baseline"}, "theriad", "later observations")
    assert len(calls) == 1
    assert calls[0].prompt == "initial prompt"
    assert session.resolve_model("planner", "baseline", "theriad") == MODEL
    with pytest.raises(EpisodeRoutingError, match="Baseline configuration changed"):
        session.resolve_model("planner", "changed", "theriad")
    with pytest.raises(EpisodeRoutingError, match="provider"):
        session.resolve_model("planner", "baseline", "openai")
    with pytest.raises(EpisodeRoutingError, match="new role"):
        session.resolve_model("new_role", "baseline", "theriad")


def test_nested_context_restores_after_failure():
    outer, inner = EpisodeRoutingSession("outer"), EpisodeRoutingSession("inner")
    with episode_routing(outer):
        with pytest.raises(ValueError):
            with episode_routing(inner):
                assert current_episode() is inner
                raise ValueError("stop")
        with episode_routing(None):
            assert current_episode() is outer
    assert current_episode() is None


@pytest.mark.parametrize("kwargs", [
    {"selected_models": {"*": MODEL}},
    {"selected_models": {"*": MODEL}, "model_metadata": METADATA, "model_filter": lambda *_: False},
    {"selected_models": {"*": MODEL}, "model_metadata": METADATA, "free_only": True},
])
def test_missing_capabilities_account_and_free_constraints_fail_closed(kwargs):
    session = EpisodeRoutingSession("e", **kwargs)
    with pytest.raises(EpisodeRoutingError):
        session.prepare("search", {"search_query": "baseline"}, "theriad")
    assert not session.attempts


def test_unknown_free_pricing_rejected_and_verified_free_allowed():
    model = "fixture/unknown:free"
    metadata = {model: {"capabilities": ["text"]}}
    session = EpisodeRoutingSession("e", selected_models={"*": model}, model_metadata=metadata, free_only=True)
    with pytest.raises(EpisodeRoutingError, match="pricing is unknown"):
        session.prepare("search", {"query": "baseline"}, "theriad")
    metadata[model].update(input_per_1m=0, output_per_1m=0)
    session = EpisodeRoutingSession("e2", selected_models={"*": model}, model_metadata=metadata, free_only=True)
    session.prepare("search", {"query": "baseline"}, "theriad")
    assert session.resolve_model("query", "baseline", "theriad") == model


@pytest.mark.asyncio
async def test_started_event_precedes_provider_and_pending_is_unknown(tmp_path):
    path = tmp_path / "ledger.jsonl"
    session = EpisodeRoutingSession("e", sink=JsonlAttemptSink(path))

    class CheckedProvider(Provider):
        async def complete(self, req):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            assert events[-1]["event"] == "attempt_started"
            assert recover_attempts(events)[0]["cost_known"] is False
            return await super().complete(req)

    with episode_routing(session):
        await complete_with_telemetry(CheckedProvider(), request())
    pending_id = session.start_attempt(kind="model", model=MODEL)
    summary = session.summary()
    assert summary["pending_ids"] == [pending_id]
    assert summary["all_costs_known"] is False
    assert summary["total_cost_usd"] is None
    assert "private input" not in path.read_text()


@pytest.mark.asyncio
async def test_failure_retry_keeps_unknown_cost_and_frozen_response_price():
    session = EpisodeRoutingSession("e", model_metadata=METADATA)
    provider = Provider([TimeoutError("private error content"), "done"])
    with episode_routing(session):
        with pytest.raises(TimeoutError):
            await complete_with_telemetry(provider, request())
        await complete_with_telemetry(provider, request())
    assert len(session.attempts) == 2
    assert session.attempts[0]["status"] == "timeout"
    assert session.attempts[0]["cost_known"] is False
    assert session.attempts[1]["cost_usd"] == pytest.approx(0.0004)
    assert session.attempts[1]["reasoning_tokens"] == 12
    assert session.summary()["failed_count"] == 1
    assert session.summary()["total_cost_usd"] is None
    assert session.summary()["known_total_cost_usd"] == pytest.approx(0.0004)
    assert "private error content" not in json.dumps(session.events)


@pytest.mark.asyncio
async def test_hard_caps_stop_before_dispatch_and_cap_output_without_mutating_request():
    session = EpisodeRoutingSession("e", max_model_calls=1, max_input_tokens=200, max_output_tokens=12)
    provider = Provider()
    original = request(max_tokens=1000)
    with episode_routing(session):
        await complete_with_telemetry(provider, original)
        with pytest.raises(EpisodeRoutingError, match="model-call limit"):
            await complete_with_telemetry(provider, original)
    assert original.max_tokens == 1000
    assert provider.requests[0].max_tokens == 12
    assert len(session.attempts) == len(provider.requests) == 1
    session = EpisodeRoutingSession("other", max_input_tokens=1)
    with episode_routing(session), pytest.raises(EpisodeRoutingError, match="input-token"):
        await complete_with_telemetry(provider, original)
    assert not session.events


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,content", [
    ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "input"),
    ({}, [{"type": "image_url", "image_url": {"url": "https://private.invalid/image"}}]),
])
async def test_actual_native_tools_and_vision_require_metadata(extra, content):
    session = EpisodeRoutingSession("e", selected_models={"*": MODEL}, model_metadata=METADATA)
    session.prepare("chat", {"chat": "baseline"}, "theriad")
    provider = Provider()
    req = CompletionRequest(model=MODEL, messages=[Message("user", content)], extra=extra)
    with episode_routing(session), pytest.raises(EpisodeRoutingError, match="capabilities"):
        await complete_with_telemetry(provider, req)
    assert not provider.requests
    assert not session.attempts


@pytest.mark.asyncio
async def test_accounting_sink_failure_poisons_session_and_never_retries_paid_call():
    def broken(_event):
        raise OSError("private filesystem details")

    session = EpisodeRoutingSession("e", sink=broken)
    provider = Provider()
    with episode_routing(session):
        for _ in range(2):
            with pytest.raises(EpisodeRoutingError, match="accounting"):
                await complete_with_telemetry(provider, request())
    assert not provider.requests


@pytest.mark.asyncio
async def test_cancellation_and_rejected_responses_are_durable():
    session = EpisodeRoutingSession("e", model_metadata=METADATA)
    with episode_routing(session):
        with pytest.raises(asyncio.CancelledError):
            await complete_with_telemetry(Provider([asyncio.CancelledError()]), request())
        await complete_with_telemetry(Provider(), request())
        session.reject_response("invalid_json")
    records = recover_attempts(session.events)
    assert records[0]["status"] == "cancelled"
    assert records[1]["response_rejected"] is True
    assert records[1]["cost_usd"] == pytest.approx(0.0004)


@pytest.mark.asyncio
async def test_real_task_planner_json_retry_is_pinned_and_charged(monkeypatch, tmp_path):
    from theriad.agents.task_engine import TaskEngine
    from theriad.agents.task_runner import run_task

    engine = TaskEngine(event_dir=tmp_path / "events")
    engine.allowed_workspace_root = tmp_path.resolve()
    task = engine.create(goal="finish", workspace_root=str(tmp_path), permission_profile="autonomous")
    provider = Provider([RuntimeError("JSON mode unavailable"), '{"tool":"task.complete","arguments":{"answer":"done"}}'])
    monkeypatch.setenv("THERIAD_TEST_LIVE_MODELS", "true")
    monkeypatch.setattr("theriad.core.providers.factory.build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr("theriad.settings.secrets.get_active_assistant_provider", lambda: "theriad")
    session = EpisodeRoutingSession("e", selected_models={"*": MODEL}, model_metadata=METADATA,
                                    max_model_calls=2, tool_prices={"task.complete": 0.0})
    result = await run_task(task.task_id, engine=engine, routing_session=session)
    assert result["status"] == "completed"
    assert result["final_answer"] == "done"
    assert [item.model for item in provider.requests] == [MODEL, MODEL]
    assert "response_format" in provider.requests[0].extra
    assert "response_format" not in provider.requests[1].extra
    assert [item["role"] for item in session.attempts] == ["agent_planner", "agent_planner_json_retry", "task.complete"]
    assert session.summary()["all_costs_known"] is False
    assert current_episode() is None


@pytest.mark.asyncio
async def test_tool_cost_unknown_by_default_and_known_from_fixture_price():
    async def operation():
        return "private tool result"

    session = EpisodeRoutingSession("e", tool_prices={"fixture.read": 0.0})
    assert await session.run_tool("fixture.read", operation) == "private tool result"
    assert session.summary()["total_cost_usd"] == 0.0
    await session.run_tool("web.search", operation)
    assert session.summary()["total_cost_usd"] is None
    assert "private tool result" not in json.dumps(session.events)


@pytest.mark.asyncio
async def test_real_search_path_routes_query_and_synthesis_once_with_frozen_evidence(monkeypatch):
    from theriad.agents.search_synthesis import synthesize_web_search
    from theriad.search.service import SearchResult

    class FrozenSearch:
        async def search(self, query):
            return [SearchResult("Fixture", "https://fixture.invalid/source", "Answer is 42.", "fixture")]

        async def fetch(self, url):
            return {"url": url, "title": "Fixture", "text": "Answer is 42.", "status": "ok"}

    provider = Provider(['{"queries": []}', "The answer is 42. [1]"])
    monkeypatch.setenv("THERIAD_TEST_LIVE_MODELS", "true")
    monkeypatch.setenv("THERIAD_API_SEARCH_SYNTHESIS", "true")
    monkeypatch.setattr("theriad.core.providers.factory.build_provider", lambda *_a, **_kw: provider)
    monkeypatch.setattr("theriad.settings.secrets.get_active_assistant_provider", lambda: "theriad")
    decisions = []

    def resolver(context):
        decisions.append(context)
        return {"*": MODEL}

    session = EpisodeRoutingSession("e", resolver=resolver, model_metadata=METADATA,
        max_model_calls=2, max_output_tokens=250, tool_prices={"web.search": 0, "web.fetch": 0})
    answer, results, queries = await synthesize_web_search(
        "What is the answer?", service=FrozenSearch(), routing_session=session)
    assert "42" in answer
    assert len(results) == len(queries) == len(decisions) == 1
    assert set(decisions[0].baseline_models) == {"search_query", "search_synthesis"}
    assert [req.model for req in provider.requests] == [MODEL, MODEL]
    assert [req.max_tokens for req in provider.requests] == [220, 250]
    assert session.summary()["all_costs_known"] is True
    assert session.summary()["attempt_count"] == 4
    assert current_episode() is None


@pytest.mark.asyncio
async def test_browser_entrypoint_prepares_all_actual_roles_before_advancing(monkeypatch):
    from types import SimpleNamespace

    from theriad.agents import browser_runtime

    seen = []
    async def loop(browser, *, mock_mode):
        seen.append((current_episode(), browser.goal, mock_mode))

    monkeypatch.setattr(browser_runtime, "_run_until_pause", loop)
    monkeypatch.setattr("theriad.settings.secrets.get_active_assistant_provider", lambda: "theriad")
    monkeypatch.setattr("theriad.agents.browser_session_store.get_session_store",
                        lambda: SimpleNamespace(checkpoint_sync=lambda _session: None))
    browser = SimpleNamespace(goal="Find the fixture", model_usage=[])
    session = EpisodeRoutingSession("e", selected_models={"*": MODEL}, model_metadata=METADATA)
    await browser_runtime.run_until_pause(browser, mock_mode=False, routing_session=session)
    assert seen == [(session, browser.goal, False)]
    prepared = session.events[0]
    assert len(prepared["baseline_models"]) == 5
    assert set(prepared["selected_models"].values()) == {MODEL}
    assert current_episode() is None


def test_free_failure_bound_requires_all_prices_zero_and_no_paid_overrides():
    from theriad.routing.episode_hooks import zero_charge_failure_bound
    quote = {'input_per_1m': 0, 'output_per_1m': 0,
             'provider_pricing': {'prompt': '0', 'completion': '0', 'request': '0'}}
    assert zero_charge_failure_bound('test/model:free', 'openrouter', quote, {}) == 0
    assert zero_charge_failure_bound('test/model', 'openrouter', quote, {}) is None
    assert zero_charge_failure_bound('test/model:free', 'other', quote, {}) is None
    assert zero_charge_failure_bound('test/model:free', 'openrouter', {}, {}) is None
    for extra in ({'models': ['paid/model']}, {'plugins': []}, {'tools': []}):
        assert zero_charge_failure_bound('test/model:free', 'openrouter', quote, extra) is None
    for rate in ('0.01', 'NaN', False, None):
        bad = {**quote, 'provider_pricing': {**quote['provider_pricing'], 'request': rate}}
        assert zero_charge_failure_bound('test/model:free', 'openrouter', bad, {}) is None


@pytest.mark.asyncio
async def test_zero_price_failure_is_failed_and_unbilled_with_unknown_usage(tmp_path):
    from theriad.core.providers.interfaces import ProviderError
    model = 'fixture/model:free'
    quote = {'capabilities': ['text'], 'input_per_1m': 0, 'output_per_1m': 0,
             'provider_pricing': {'prompt': '0', 'completion': '0'}}
    session = EpisodeRoutingSession('free-failure', model_metadata={model: quote}, sink=JsonlAttemptSink(tmp_path / 'attempts.jsonl'))
    session.prepare('chat', {'model': model}, 'theriad', 'test')
    provider = Provider([ProviderError('openrouter', 'test failure', 429, request_id='request-123')])
    with episode_routing(session), pytest.raises(ProviderError):
        await complete_with_telemetry(provider, request(model))
    attempt = session.attempts[-1]
    assert attempt['status'] == 'failed'
    assert attempt['cost_usd'] == 0
    assert attempt['billed_cost_usd'] is None
    assert attempt['usage_available'] is False
    assert attempt['provider_status_code'] == 429
    assert attempt['provider_request_id'] == 'request-123'
    assert attempt['cost_basis'] == 'frozen_zero_charge_upper_bound'
    assert all(event['recorded_at'] for event in session.events)
