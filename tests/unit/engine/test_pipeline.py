"""Tests for the TheriadPipeline execution engine."""
from __future__ import annotations

import pytest

from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderProtocol,
)
from theriad.core.routing.interfaces import RoutingDecision, RoutingLayer, RoutingResult
from theriad.engine.pipeline import TheriadPipeline
from theriad.engine.policies import ExecutionPolicy, FallbackPolicy, RetryPolicy

# ── Helpers / stubs ───────────────────────────────────────────────────────────

def make_response(content: str = "ok") -> CompletionResponse:
    return CompletionResponse(
        model="qwen/qwen3-8b", content=content,
        prompt_tokens=10, completion_tokens=5,
        latency_ms=100.0, provider="fake",
    )


class FixedLayer(RoutingLayer):
    """Layer that always routes to a fixed tier with fixed confidence."""
    def __init__(self, tier: str, confidence: float):
        self._tier = tier
        self._confidence = confidence

    @property
    def name(self) -> str:
        return "fixed_layer"

    def route(self, prompt: str, context: dict) -> RoutingResult | None:
        if self._confidence == 0:
            return None
        return RoutingResult(
            decision=RoutingDecision(tier=self._tier, confidence=self._confidence, reason="fixed"),
            layer_name=self.name,
        )


class AlwaysAbstainLayer(RoutingLayer):
    @property
    def name(self) -> str:
        return "abstain"

    def route(self, prompt, context):
        return None


class FakeProvider(ProviderProtocol):
    def __init__(self, name: str = "fake", response: CompletionResponse | None = None, raises=None):
        self._name = name
        self._response = response or make_response()
        self._raises = raises
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        if self._raises:
            raise self._raises
        return self._response

    async def health_check(self) -> bool:
        return True


# ── Routing ───────────────────────────────────────────────────────────────────

def test_routing_uses_first_layer():
    layer = FixedLayer(tier="cheap_bulk", confidence=0.9)
    pipeline = TheriadPipeline(
        layers=[layer],
        providers={},
        fallback_tier="strong_reasoning",
        fallback_provider="fake",
    )
    result = pipeline.route("hello")
    assert result.decision.tier == "cheap_bulk"
    assert result.layer_name == "fixed_layer"


def test_routing_falls_back_when_all_abstain():
    pipeline = TheriadPipeline(
        layers=[AlwaysAbstainLayer()],
        providers={},
        fallback_tier="cheap_bulk",
        fallback_provider="fake",
    )
    result = pipeline.route("hello")
    assert result.decision.tier == "cheap_bulk"
    assert result.layer_name == "fallback"


def test_routing_skips_to_second_layer_if_first_abstains():
    first = AlwaysAbstainLayer()
    second = FixedLayer(tier="strong_reasoning", confidence=0.95)
    pipeline = TheriadPipeline(layers=[first, second], providers={})
    result = pipeline.route("complex prompt")
    assert result.decision.tier == "strong_reasoning"
    assert result.layer_name == "fixed_layer"


# ── Execution ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_success():
    provider = FakeProvider(name="openrouter")
    pipeline = TheriadPipeline(
        layers=[FixedLayer("cheap_bulk", 0.9)],
        providers={"openrouter": provider},
        fallback_provider="openrouter",
    )
    messages = [Message(role="user", content="hello")]
    resp = await pipeline.execute(messages)
    assert resp.content == "ok"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_execute_retries_on_failure():
    err = ProviderError("fake", "server error", 500)
    provider = FakeProvider(name="fake", raises=err)
    policy = ExecutionPolicy(
        retry=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0),
        fallback=FallbackPolicy(chain=[]),
    )
    pipeline = TheriadPipeline(
        layers=[],
        providers={"fake": provider},
        fallback_provider="fake",
        policy=policy,
    )
    # Pass model + provider_name explicitly to bypass tier resolution
    # (these tests exercise retry/fallback logic, not routing)
    with pytest.raises(ProviderError):
        await pipeline.execute(
            [Message(role="user", content="hi")],
            model="test-model",
            provider_name="fake",
        )
    assert provider.call_count == 3  # 3 retries


@pytest.mark.asyncio
async def test_execute_does_not_retry_4xx():
    err = ProviderError("fake", "bad request", 400)
    provider = FakeProvider(name="fake", raises=err)
    policy = ExecutionPolicy(retry=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0))
    pipeline = TheriadPipeline(
        layers=[],
        providers={"fake": provider},
        fallback_provider="fake",
        policy=policy,
    )
    with pytest.raises(ProviderError) as exc_info:
        await pipeline.execute(
            [Message(role="user", content="hi")],
            model="test-model",
            provider_name="fake",
        )
    assert exc_info.value.status_code == 400
    assert provider.call_count == 1  # no retry on 4xx


@pytest.mark.asyncio
async def test_execute_uses_fallback_chain():
    fail_provider = FakeProvider(name="primary", raises=ProviderError("primary", "down", 500))
    ok_provider = FakeProvider(name="backup")
    policy = ExecutionPolicy(
        retry=RetryPolicy(max_attempts=1, backoff_base_seconds=0.0),
        fallback=FallbackPolicy(chain=[("backup", "backup-model")]),
    )
    pipeline = TheriadPipeline(
        layers=[],
        providers={"primary": fail_provider, "backup": ok_provider},
        fallback_provider="primary",
        policy=policy,
    )
    resp = await pipeline.execute([Message(role="user", content="hi")])
    assert resp.content == "ok"
    assert ok_provider.call_count == 1


@pytest.mark.asyncio
async def test_execute_raises_when_all_exhausted():
    err = ProviderError("p", "fail", 500)
    p1 = FakeProvider(name="p1", raises=err)
    p2 = FakeProvider(name="p2", raises=err)
    policy = ExecutionPolicy(
        retry=RetryPolicy(max_attempts=1, backoff_base_seconds=0.0),
        fallback=FallbackPolicy(chain=[("p2", "model2")]),
    )
    pipeline = TheriadPipeline(
        layers=[],
        providers={"p1": p1, "p2": p2},
        fallback_provider="p1",
        policy=policy,
    )
    with pytest.raises(ProviderError, match="all providers exhausted"):
        await pipeline.execute([Message(role="user", content="hi")])
