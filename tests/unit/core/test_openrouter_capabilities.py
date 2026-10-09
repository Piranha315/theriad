from copy import deepcopy
from datetime import datetime, timedelta, timezone
import asyncio
import json

import httpx
import pytest

from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError
from theriad.core.providers.openrouter import OpenRouterProvider
from theriad.core.providers.openrouter_capabilities import LUNA, SNAPSHOT_KEY, catalog_digest, compatible_payload
from theriad.routing.study_adapters import StructuredCompletionRequest


def snapshot():
    endpoint = {"tag": "openai/flex", "status": 0, "max_completion_tokens": 128000,
                "supported_parameters": ["max_tokens", "tools", "tool_choice", "response_format", "seed"],
                "supports_tool_choice": {"auto": True, "required": True, "function": True}}
    catalog = {"data": {"id": LUNA, "endpoints": [endpoint,
        {**endpoint, "tag": "azure/us", "status": -5},
        {**endpoint, "tag": "amazon-bedrock/us-east-1", "supported_parameters": ["max_tokens", "tools"]}]}}
    return {"catalog": catalog, "catalog_digest": catalog_digest(catalog),
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "endpoint_tag": "openai/flex"}


@pytest.mark.parametrize("tool_request", [False, True])
@pytest.mark.parametrize("model", [LUNA, "deepseek/deepseek-v4-flash-0731", "fixture/model"])
def test_wire_payloads(model, tool_request):
    seen = []
    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    provider = OpenRouterProvider("fixture", transport=httpx.MockTransport(transport))
    meta = {"provider": {"require_parameters": True}, "seed": 7}
    if model == LUNA:
        meta[SNAPSHOT_KEY] = snapshot()
    original = deepcopy(meta)
    if tool_request:
        req = StructuredCompletionRequest(model=model, messages=(Message("user", "hello"),),
            tool_schemas=({"name": "lookup", "parameters": {"type": "object"}},),
            temperature=0, max_tokens=128, metadata=meta)
        asyncio.run(provider.complete_with_tools(req))
    else:
        asyncio.run(provider.complete(CompletionRequest(model, [Message("user", "hello")], max_tokens=128, extra=meta)))
    assert meta == original
    body = seen[0]
    assert SNAPSHOT_KEY not in body
    assert body["seed"] == 7 and body["max_tokens"] == 128
    assert body["provider"]["require_parameters"] is True
    if model == LUNA:
        assert "temperature" not in body
        assert body["provider"] == {"require_parameters": True, "only": ["openai/flex"], "order": ["openai/flex"], "allow_fallbacks": False}
    else:
        assert body["temperature"] == 0
        assert body["provider"] == {"require_parameters": True}
    assert ("tools" in body) == tool_request


@pytest.mark.parametrize("failure", ["missing", "stale", "future", "digest", "model", "alias", "absent", "unhealthy", "required_format", "output", "require_false", "conflict", "fallback", "tool_choice"])
@pytest.mark.parametrize("tool_request", [False, True])
def test_preflight_rejects_without_dispatch(failure, tool_request):
    s = snapshot()
    meta = {SNAPSHOT_KEY: s}
    if failure == "missing": meta.clear()
    if failure == "stale": s["retrieved_at"] = (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()
    if failure == "future": s["retrieved_at"] = (datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    if failure == "digest": s["catalog_digest"] = "bad"
    if failure == "model":
        s["catalog"]["data"]["id"] = "wrong"
        s["catalog_digest"] = catalog_digest(s["catalog"])
    if failure == "alias": s["endpoint_tag"] = "openai"
    if failure == "absent": s["endpoint_tag"] = "openai/missing"
    if failure == "unhealthy": s["endpoint_tag"] = "azure/us"
    if failure == "required_format":
        s["endpoint_tag"] = "amazon-bedrock/us-east-1"
        meta["seed"] = 4
    if failure == "require_false": meta["provider"] = {"require_parameters": False}
    if failure == "conflict": meta["provider"] = {"only": ["openai"]}
    if failure == "fallback": meta["models"] = [LUNA, "fixture/model"]
    if failure == "tool_choice": meta["tool_choice"] = "none"
    def transport(request):
        pytest.fail("Rejected request reached HTTP")
    p = OpenRouterProvider("fixture", transport=httpx.MockTransport(transport))
    tokens = 128001 if failure == "output" else 128
    with pytest.raises(ProviderError, match="capability preflight"):
        if tool_request:
            asyncio.run(p.complete_with_tools(StructuredCompletionRequest(model=LUNA,
                messages=(Message("user", "hello"),), tool_schemas=(), max_tokens=tokens, metadata=meta)))
        else:
            asyncio.run(p.complete(CompletionRequest(LUNA, [Message("user", "hello")], max_tokens=tokens, extra=meta)))


def test_unsupported_sampling_omitted_but_semantics_fail_closed():
    meta = {SNAPSHOT_KEY: snapshot()}
    body = {"model": LUNA, "messages": [], "temperature": 0, "top_p": .8,
            "frequency_penalty": 0, "presence_penalty": .1, "max_tokens": 10}
    result = compatible_payload(body, meta)
    assert not {"temperature", "top_p", "frequency_penalty", "presence_penalty"} & result.keys()
    for key in ("stop", "parallel_tool_calls", "unknown_semantic_constraint"):
        with pytest.raises(ProviderError):
            compatible_payload({**body, key: True}, meta)


def test_healthy_endpoint_with_smaller_feature_set_is_eligible_only_when_sufficient():
    s = snapshot(); s["endpoint_tag"] = "amazon-bedrock/us-east-1"
    body = {"model": LUNA, "messages": [], "temperature": 0, "max_tokens": 10, "tools": []}
    assert compatible_payload(body, {SNAPSHOT_KEY: s})["provider"]["only"] == ["amazon-bedrock/us-east-1"]
    with pytest.raises(ProviderError):
        compatible_payload({**body, "response_format": {"type": "json_object"}}, {SNAPSHOT_KEY: s})


def test_real_factory_and_episode_guard_preserve_snapshot_without_token_inflation(monkeypatch):
    from theriad.core.providers.factory import build_provider
    from theriad.routing.episode_hooks import EpisodeRoutingSession, episode_routing
    from theriad.agents.model_telemetry import complete_with_telemetry
    seen = []
    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": LUNA, "provider": "OpenAI", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"cost": .001}})
    provider = build_provider("openrouter", api_key="fixture")
    provider._transport = httpx.MockTransport(transport)
    session = EpisodeRoutingSession("fixture", model_metadata={LUNA: {SNAPSHOT_KEY: snapshot()}}, max_input_tokens=100)
    with episode_routing(session):
        asyncio.run(complete_with_telemetry(provider, CompletionRequest(LUNA, [Message("user", "hello")], max_tokens=10)))
    assert seen[0]["messages"][-1]["content"] == "hello"
    assert "temperature" not in seen[0] and SNAPSHOT_KEY not in seen[0]
    assert session.attempts[-1]["cost_usd"] == .001


def test_local_preflight_rejection_is_known_zero_not_another_billing_hole():
    from theriad.routing.episode_hooks import EpisodeRoutingSession, episode_routing
    from theriad.agents.model_telemetry import complete_with_telemetry
    provider = OpenRouterProvider("fixture", transport=httpx.MockTransport(lambda _: pytest.fail("HTTP dispatched")))
    session = EpisodeRoutingSession("fixture")
    with episode_routing(session), pytest.raises(ProviderError):
        asyncio.run(complete_with_telemetry(provider, CompletionRequest(LUNA, [Message("user", "hello")], max_tokens=10)))
    assert session.attempts[-1]["cost_usd"] == 0
    assert session.attempts[-1]["request_dispatched"] is False
    assert session.attempts[-1]["cost_basis"] == "local_capability_rejection_no_dispatch"


def test_other_model_with_null_optional_fallbacks_is_unchanged():
    body = {"model": "fixture/model", "models": None, "temperature": .4, "messages": []}
    assert compatible_payload(body, {}) is body
