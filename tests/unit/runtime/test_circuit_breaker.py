from __future__ import annotations

import time

import pytest

from theriad.runtime.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError


def test_circuit_opens_then_allows_one_half_open_probe():
    registry = CircuitBreakerRegistry(failure_threshold=2, cooldown_seconds=0.02)
    registry.record_failure("search:test", RuntimeError("one"))
    registry.record_failure("search:test", RuntimeError("two"))

    with pytest.raises(CircuitOpenError):
        registry.before_call("search:test")

    time.sleep(0.03)
    registry.before_call("search:test")
    with pytest.raises(CircuitOpenError):
        registry.before_call("search:test")

    registry.record_success("search:test", latency_ms=1.5)
    registry.before_call("search:test")
    row = registry.snapshot()[0]
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0


def test_non_transient_failure_does_not_open_circuit():
    registry = CircuitBreakerRegistry(failure_threshold=1, cooldown_seconds=10)
    registry.record_failure("llm:test", ValueError("bad request"), transient=False)
    registry.before_call("llm:test")
    assert registry.snapshot()[0]["status"] == "degraded"


@pytest.mark.asyncio
async def test_provider_readiness_wrapper_records_completion_failure_and_recovery():
    from theriad.core.providers.interfaces import CompletionRequest, ProviderError
    from theriad.core.providers.readiness import with_provider_readiness
    from theriad.runtime.circuit_breaker import get_circuit_registry

    class Provider:
        name = "readiness-test"

        def __init__(self):
            self.fail = True

        async def complete(self, request):
            if self.fail:
                raise ProviderError(self.name, "unavailable", status_code=503)
            return request

        async def health_check(self):
            return not self.fail

    registry = get_circuit_registry()
    registry.reset("llm:readiness-test")
    provider = with_provider_readiness(Provider())
    request = CompletionRequest(model="test", messages=[])
    with pytest.raises(ProviderError):
        await provider.complete(request)
    assert registry.snapshot("llm:readiness-test")[0]["status"] == "degraded"

    provider.fail = False
    assert await provider.health_check() is True
    assert registry.snapshot("llm:readiness-test")[0]["status"] == "ready"
    registry.reset("llm:readiness-test")
