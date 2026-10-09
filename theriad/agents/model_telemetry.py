"""Small, provider-neutral ledger for model calls made during one task.

The browser runtime and the direct web-research path use different orchestration
code, but they should expose the same facts to the user: which models actually
returned a response, how many tokens they used, and whether the cost is known.
Dispatch wrappers also record failed attempts separately from model responses.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from theriad.accounting.actuals import provider_reported_cost, realized_cost, usage_is_available
from theriad.accounting.pricing import PricingTable

_TRUNCATION_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "MAX_TOKENS", "max_output_tokens"}
)


def is_truncated_finish_reason(finish_reason: Any) -> bool:
    """Return whether a provider stopped because its output budget was hit."""
    value = str(finish_reason or "").strip()
    return value in _TRUNCATION_FINISH_REASONS


_MODEL_USAGE_SINK: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "theriad_model_usage_sink", default=None
)


@contextmanager
def capture_model_usage(sink: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    """Capture provider responses issued inside the current async task."""
    token = _MODEL_USAGE_SINK.set(sink)
    try:
        yield sink
    finally:
        _MODEL_USAGE_SINK.reset(token)


def record_model_usage(
    response: Any,
    sink: list[dict[str, Any]] | None = None,
    *, pricing_table: PricingTable | None = None,
) -> dict[str, Any]:
    """Append one response's safe accounting fields and return the entry."""
    target = sink if sink is not None else _MODEL_USAGE_SINK.get()
    model = str(getattr(response, "model", "") or "unknown")
    provider = str(getattr(response, "provider", "") or "unknown")
    prompt_tokens = max(0, int(getattr(response, "prompt_tokens", 0) or 0))
    completion_tokens = max(0, int(getattr(response, "completion_tokens", 0) or 0))
    latency_ms = float(getattr(response, "latency_ms", 0.0) or 0.0)
    raw_finish_reason = getattr(response, "finish_reason", None)
    finish_reason = str(raw_finish_reason).strip() if raw_finish_reason else None

    cost: float | None = None
    try:
        # The offline seed table is intentionally used here. It avoids making
        # a pricing lookup network call for every completion while still
        # remaining conservative for models with known list prices. A provider
        # reported billed cost always wins inside realized_cost().
        from theriad.routing.episode_hooks import current_episode
        session = current_episode()
        pricing = pricing_table or PricingTable(use_live=False)
        cost = session.price_response(response, pricing) if session else realized_cost(response, pricing)
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            cost = None
    except Exception:
        # Accounting must never turn a successful answer into a failed task.
        cost = None

    entry: dict[str, Any] = {
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(max(0.0, latency_ms), 1),
        "cost_usd": cost,
        "cost_known": cost is not None,
        "finish_reason": finish_reason,
        "truncated": is_truncated_finish_reason(finish_reason),
        "billed_cost_usd": provider_reported_cost(response),
        "usage_available": usage_is_available(response),
        "status": "truncated" if is_truncated_finish_reason(finish_reason) else "success",
    }
    raw = getattr(response, "raw", {}) or {}
    entry["generation_id"] = raw.get("id") if isinstance(raw, dict) and isinstance(raw.get("id"), str) else None
    entry["backend_provider"] = raw.get("provider") if isinstance(raw, dict) and isinstance(raw.get("provider"), str) else None
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    details = usage.get("completion_tokens_details", usage.get("output_tokens_details", {})) if isinstance(usage, dict) else {}
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    entry["reasoning_tokens"] = reasoning_tokens if isinstance(reasoning_tokens, int) and reasoning_tokens >= 0 else None
    headroom = raw.get("_theriad_headroom", {}) if isinstance(raw, dict) else {}
    if isinstance(headroom, dict):
        entry["headroom"] = {k: headroom[k] for k in ("enabled", "mode", "bypassed", "reason", "tokens_saved", "protection_restored") if k in headroom}
    if target is not None:
        target.append(entry)
    return entry


async def complete_with_telemetry(
    provider: Any, request: Any, *, role: str = "model",
    usage_sink: list[dict[str, Any]] | None = None,
    pricing_table: PricingTable | None = None,
) -> Any:
    """Record every dispatched completion, including failures before a retry.

    A session's start event is durable before the provider call. Errors retain
    unknown charges because an upstream may have billed a timed-out request.
    The ledger stores exception types only; messages may contain private input.
    """
    from theriad.routing.episode_hooks import current_episode

    session = current_episode()
    if session is not None:
        request = session.guard_request(request)
    provider_name = getattr(provider, "name", None)
    provider_name = provider_name if isinstance(provider_name, str) else None
    attempt_id = session.start_attempt(
        kind="model", model=request.model, provider=provider_name, role=role,
    ) if session else None
    started = time.perf_counter()
    try:
        response = await provider.complete(request)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else (
            "timeout" if isinstance(exc, TimeoutError) else "failed"
        )
        failure_cost = session.failed_request_cost_bound(request, provider_name) if session else None
        from theriad.runtime.circuit_breaker import CircuitOpenError
        from theriad.core.providers.openrouter_capabilities import CapabilityPreflightError
        local_rejection = isinstance(exc, (CircuitOpenError, CapabilityPreflightError))
        if local_rejection:
            failure_cost = 0.0
        entry = {
            "kind": "model", "model": request.model, "provider": provider_name,
            "role": role, "status": status, "error_type": type(exc).__name__,
            "prompt_tokens": None, "completion_tokens": None,
            "cost_usd": failure_cost, "cost_known": failure_cost is not None, "billed_cost_usd": None,
            "cost_basis": "local_capability_rejection_no_dispatch" if isinstance(exc, CapabilityPreflightError) else "local_circuit_rejection_no_dispatch" if local_rejection else "frozen_zero_charge_upper_bound" if failure_cost is not None else "unknown",
            "request_dispatched": False if local_rejection else None,
            "provider_status_code": getattr(exc, "status_code", None),
            "provider_request_id": getattr(exc, "request_id", None),
            "usage_available": False, "finish_reason": None, "truncated": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        target = usage_sink if usage_sink is not None else _MODEL_USAGE_SINK.get()
        if target is not None:
            target.append(entry)
        if session and attempt_id:
            session.finish_attempt(attempt_id, entry)
        if session and getattr(session, "decision_resolver", None) is not None:
            session.mark_model_unavailable(request.model)
        raise
    entry = record_model_usage(response, usage_sink, pricing_table=pricing_table)
    entry.update({"kind": "model", "role": role,
                  "latency_ms": (time.perf_counter() - started) * 1000})
    if session and getattr(session, 'response_sink', None):
        session.response_sink({'episode_id': session.episode_id, 'policy_id': session.policy_id,
            'attempt_id': attempt_id, 'role': role, 'model': request.model,
            'content': response.content, 'reasoning': getattr(response, 'reasoning', None),
            'finish_reason': getattr(response, 'finish_reason', None),
            'request_max_tokens': request.max_tokens, 'usage': entry})
    if session and attempt_id:
        session.finish_attempt(attempt_id, entry)
    if session:
        session.verify_response_backend(request.model, entry.get("backend_provider"))
    return response


def summarize_model_usage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the response ledger by provider/model for a compact API trace."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry.get("provider") or "unknown"), str(entry.get("model") or "unknown"))
        current = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "model": key[1],
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
                "cost_known": True,
                "finish_reasons": [],
                "truncated": False,
            },
        )
        current["calls"] += 1
        current["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        current["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        current["latency_ms"] += float(entry.get("latency_ms") or 0.0)
        if entry.get("cost_usd") is None:
            current["cost_known"] = False
        else:
            current["cost_usd"] += float(entry["cost_usd"])
        finish_reason = entry.get("finish_reason")
        if finish_reason and finish_reason not in current["finish_reasons"]:
            current["finish_reasons"].append(finish_reason)
        current["truncated"] = bool(current["truncated"] or entry.get("truncated"))

    output = []
    for item in grouped.values():
        item["latency_ms"] = round(item["latency_ms"], 1)
        item["cost_usd"] = round(item["cost_usd"], 8) if item["cost_known"] else None
        output.append(item)
    return output


def usage_totals(entries: list[dict[str, Any]]) -> tuple[float | None, bool]:
    """Return (total cost, cost_known) without pretending unknown is free."""
    if not entries:
        return 0.0, True
    if any(entry.get("cost_usd") is None for entry in entries):
        return None, False
    return round(sum(float(entry["cost_usd"]) for entry in entries), 8), True
