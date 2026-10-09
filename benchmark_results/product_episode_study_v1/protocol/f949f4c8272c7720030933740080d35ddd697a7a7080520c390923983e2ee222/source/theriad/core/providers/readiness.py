"""Provider completion/readiness decorator with bounded circuit breaking."""
from __future__ import annotations

import time
from typing import Any

from theriad.runtime.circuit_breaker import get_circuit_registry

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol


def _transient(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    return status is None or status in {408, 409, 425, 429} or (
        isinstance(status, int) and status >= 500
    )


def with_provider_readiness(provider: ProviderProtocol) -> ProviderProtocol:
    """Decorate a provider once while preserving its concrete type."""
    if getattr(provider, "_theriad_readiness_wrapped", False):
        return provider
    original_complete = getattr(provider, "complete", None)
    original_health_check = getattr(provider, "health_check", None)
    # Provider registration/import tests may use constructor-only sentinels.
    # A live adapter always implements the protocol, but decoration should not
    # make dynamic adapter discovery more brittle.
    if not callable(original_complete) or not callable(original_health_check):
        return provider
    key = f"llm:{provider.name}"

    async def complete(request: CompletionRequest) -> CompletionResponse:
        registry = get_circuit_registry()
        registry.before_call(key)
        started = time.perf_counter()
        try:
            response = await original_complete(request)
        except Exception as exc:
            registry.record_failure(
                key,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
                transient=_transient(exc),
            )
            raise
        registry.record_success(
            key,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    async def health_check() -> bool:
        registry = get_circuit_registry()
        started = time.perf_counter()
        try:
            ok = await original_health_check()
        except Exception as exc:
            registry.record_failure(
                key,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return False
        latency_ms = (time.perf_counter() - started) * 1000
        if ok:
            registry.record_success(key, latency_ms=latency_ms)
        else:
            registry.record_failure(
                key,
                RuntimeError("health check failed"),
                latency_ms=latency_ms,
            )
        return ok

    original_tools = getattr(provider, "complete_with_tools", None)

    async def complete_with_tools(request: Any) -> Any:
        if not callable(original_tools):
            raise ProviderError(provider.name, "provider does not support structured tool calls")
        registry = get_circuit_registry()
        registry.before_call(key)
        started = time.perf_counter()
        try:
            response = await original_tools(request)
        except Exception as exc:
            registry.record_failure(
                key,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
                transient=_transient(exc),
            )
            raise
        registry.record_success(
            key,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    provider.complete = complete  # type: ignore[method-assign]
    provider.health_check = health_check  # type: ignore[method-assign]
    if callable(original_tools):
        provider.complete_with_tools = complete_with_tools  # type: ignore[attr-defined]
    provider._theriad_readiness_wrapped = True  # type: ignore[attr-defined]
    return provider
