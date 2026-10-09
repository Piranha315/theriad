"""Opt-in, episode-sticky routing and durable attempt accounting.

Production callers need not create a session. Study callers supply a frozen
model configuration and a synchronous durable sink; no prompts, responses, tool
arguments, exception messages, or credentials enter this ledger. Sink failures
propagate before more paid work can be dispatched.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any


class EpisodeRoutingError(RuntimeError):
    """An opt-in configuration could not preserve the runtime restrictions."""


class JsonlAttemptSink:
    """Append and fsync each event, including the pre-dispatch reservation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        # flock also serializes separate processes writing the study ledger.
        import fcntl

        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)


@dataclass(frozen=True)
class EpisodeRoutingRequest:
    episode_id: str
    workload: str
    baseline_models: Mapping[str, str]
    assistant_provider: str
    free_only: bool
    # Only pre-execution inputs are visible to the resolver.
    prompt: str
    required_capabilities: frozenset[str]


_CURRENT_EPISODE: ContextVar[EpisodeRoutingSession | None] = ContextVar(
    "theriad_episode_routing", default=None
)


def current_episode() -> EpisodeRoutingSession | None:
    return _CURRENT_EPISODE.get()


@contextmanager
def episode_routing(session: EpisodeRoutingSession | None) -> Iterator[EpisodeRoutingSession | None]:
    """Bind a session to this async trajectory, preserving an outer binding."""
    token = _CURRENT_EPISODE.set(session or current_episode())
    try:
        yield current_episode()
    finally:
        _CURRENT_EPISODE.reset(token)


def zero_charge_failure_bound(model: str, provider: str, metadata: Mapping[str, Any], extra: Mapping[str, Any]) -> float | None:
    """A zero upper bound is provable only for an explicitly free request.

    Unknown usage is still unknown and the attempt is still failed. This is
    not a default for failed requests: every published charge must be zero,
    and model/plugin overrides that could introduce another charge are banned.
    """
    if provider != "openrouter" or not model.endswith(":free"):
        return None
    if any(key in extra for key in ("models", "plugins", "web_search_options", "route", "tools", "modalities")):
        return None
    prices = metadata.get("provider_pricing")
    if not isinstance(prices, Mapping) or not {"prompt", "completion"}.issubset(prices):
        return None
    try:
        if any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) != 0 for value in prices.values()):
            return None
        if any(isinstance(metadata.get(key), bool) or metadata.get(key) != 0 for key in ("input_per_1m", "output_per_1m")):
            return None
    except (TypeError, ValueError):
        return None
    return 0.0


class EpisodeRoutingSession:
    """One selected configuration for a whole episode, including retries.

    ``selected_models`` maps runtime roles (or ``*``) to concrete slugs. A
    resolver is called once with the full initial baseline configuration.
    Changed models require caller-supplied, frozen capability metadata; baseline
    roles retain their existing runtime behavior. Providers cannot be changed.
    """

    def __init__(
        self, episode_id: str, *, policy_id: str = "baseline", group_id: str = "",
        fixture_version: str = "", grader_version: str = "",
        selected_models: Mapping[str, str] | None = None,
        model_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        resolver: Callable[[EpisodeRoutingRequest], Mapping[str, str]] | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
        free_only: bool = False,
        model_filter: Callable[[str, str], bool] | None = None,
        max_output_tokens: int | None = None,
        tool_prices: Mapping[str, float] | None = None,
        max_model_calls: int | None = None,
        max_input_tokens: int | None = None,
        image_input_token_bound: int | None = None,
        provider_factory: Callable[..., Any] | None = None,
        assistant_provider: str | None = None,
        decision_resolver: Callable[..., str] | None = None,
        use_full_output_limit: bool = False,
        response_sink: Callable[[dict], None] | None = None,
    ):
        if not episode_id:
            raise ValueError("episode_id is required")
        if resolver is not None and selected_models is not None:
            raise ValueError("Provide a resolver or selected_models, not both")
        for value in (max_output_tokens, max_model_calls, max_input_tokens, image_input_token_bound):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError("Episode limits must be positive integers")
        self.episode_id, self.policy_id, self.group_id = episode_id, policy_id, group_id
        self.fixture_version, self.grader_version = fixture_version, grader_version
        self.free_only = free_only
        self.sink = sink
        self.max_output_tokens = max_output_tokens
        self.use_full_output_limit = use_full_output_limit
        self.response_sink = response_sink
        self.max_model_calls = max_model_calls
        self.max_input_tokens = max_input_tokens
        self.image_input_token_bound = image_input_token_bound
        self.provider_factory = provider_factory
        self.assistant_provider = assistant_provider
        self.decision_resolver = decision_resolver
        self._resolved_dispatch_model: str | None = None
        self.unavailable_models: set[str] = set()
        self._tool_prices = dict(tool_prices or {})
        if any(isinstance(value, bool) or not math.isfinite(value) or value < 0
               for value in self._tool_prices.values()):
            raise ValueError("Tool prices must be finite nonnegative charges per operation")
        self._model_filter = model_filter
        self._resolver = resolver
        self._selected = dict(selected_models or {})
        self._metadata = deepcopy(dict(model_metadata or {}))
        self._prepared = False
        self._accounting_failed = False
        self._provider: str | None = None
        self._baselines: dict[str, str] = {}
        self._required_capabilities = frozenset({"text"})
        self.attempts: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        if self._accounting_failed:
            raise EpisodeRoutingError("Durable episode accounting previously failed")
        item = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1, "episode_id": self.episode_id,
            "group_id": self.group_id, "policy_id": self.policy_id,
            "fixture_version": self.fixture_version, "grader_version": self.grader_version,
            **event,
        }
        if self.sink is not None:
            try:
                self.sink(dict(item))
            except Exception as exc:
                self._accounting_failed = True
                raise EpisodeRoutingError("Durable episode accounting failed") from exc
        self.events.append(dict(item))
        return item

    def failed_request_cost_bound(self, request: Any, provider: str | None) -> float | None:
        return zero_charge_failure_bound(request.model, provider or "", self._metadata.get(request.model, {}), request.extra or {})

    def prepare(
        self, workload: str, baseline_models: Mapping[str, str], assistant_provider: str,
        prompt: str = "", *, free_only: bool = False,
        required_capabilities: frozenset[str] = frozenset({"text"}),
    ) -> None:
        if self._prepared:
            if self._provider != assistant_provider:
                raise EpisodeRoutingError("An episode cannot cross assistant-provider boundaries")
            if free_only and not self.free_only:
                raise EpisodeRoutingError("Nested free-only execution requires a free-only episode")
            return
        started = time.perf_counter()
        self.free_only = self.free_only or free_only
        self._provider = assistant_provider
        self._baselines = dict(baseline_models)
        self._required_capabilities = required_capabilities
        if self._resolver is not None:
            self._selected = dict(self._resolver(EpisodeRoutingRequest(
                self.episode_id, workload, MappingProxyType(dict(baseline_models)),
                assistant_provider, self.free_only, prompt, required_capabilities,
            )))
        unknown_roles = set(self._selected) - set(baseline_models) - {"*"}
        if unknown_roles:
            raise EpisodeRoutingError("Configuration contains roles absent from the baseline")
        for role, baseline in baseline_models.items():
            self._validate(self._selected.get(role, self._selected.get("*", baseline)),
                           baseline, assistant_provider, required_capabilities)
        self.emit({
            "event": "episode_prepared", "workload": workload,
            "assistant_provider": assistant_provider, "free_only": self.free_only,
            "baseline_models": dict(baseline_models),
            "selected_models": {role: self._selected.get(role, self._selected.get("*", model))
                                for role, model in baseline_models.items()},
            "required_capabilities": sorted(required_capabilities),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "router_latency_ms": (time.perf_counter() - started) * 1000,
        })
        self._prepared = True

    def _validate(self, model: str, baseline: str, provider: str, required: frozenset[str]) -> None:
        if not isinstance(model, str) or not model:
            raise EpisodeRoutingError("Selected model must be a nonempty slug")
        if self._model_filter is not None and not self._model_filter(model, provider):
            raise EpisodeRoutingError("Selected model violates account/project restrictions")
        metadata = self._metadata.get(model, {})
        if model != baseline:
            if not required.issubset(set(metadata.get("capabilities") or [])):
                raise EpisodeRoutingError("Selected model lacks verified required capabilities")
            metadata_provider = metadata.get("assistant_provider", provider)
            if metadata_provider != provider:
                raise EpisodeRoutingError("Selected model belongs to another assistant provider")
        if self.free_only:
            if provider != "theriad" or not model.endswith(":free"):
                raise EpisodeRoutingError("Free-only episodes require verified free OpenRouter models")
            verified = all(
                isinstance(metadata.get(key), (int, float))
                and not isinstance(metadata.get(key), bool) and metadata[key] == 0
                for key in ("input_per_1m", "output_per_1m")
            )
            if not verified:
                from theriad.models.registry import DEFAULT_REGISTRY
                verified = any(
                    candidate.model == model and candidate.provider == "openrouter"
                    and candidate.input_cost_per_million == 0
                    and candidate.output_cost_per_million == 0
                    for tier in DEFAULT_REGISTRY.all_tiers()
                    for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier)
                )
            if not verified:
                raise EpisodeRoutingError("Free model pricing is unknown")

    def resolve_model(
        self, role: str, baseline_model: str, assistant_provider: str,
        *, required_capabilities: frozenset[str] = frozenset({"text"}),
        decision_context: str = "",
    ) -> str:
        if not self._prepared:
            raise EpisodeRoutingError("Prepare the entire configuration before a provider call")
        if assistant_provider != self._provider:
            raise EpisodeRoutingError("An episode cannot cross assistant-provider boundaries")
        expected = self._baselines.get(role)
        if expected is not None and expected != baseline_model:
            raise EpisodeRoutingError("Baseline configuration changed during the episode")
        if expected is None and self._selected and self.decision_resolver is None:
            raise EpisodeRoutingError("A new role cannot be routed midway through an episode")
        if self.decision_resolver is not None:
            selected = self.decision_resolver(
                role=role,
                baseline_model=baseline_model,
                assistant_provider=assistant_provider,
                required_capabilities=required_capabilities | self._required_capabilities,
                prompt=decision_context,
                unavailable_models=frozenset(self.unavailable_models),
            )
        else:
            selected = self._selected.get(role, self._selected.get("*", baseline_model))
        self._validate(selected, baseline_model, assistant_provider, required_capabilities | self._required_capabilities)
        if self.decision_resolver is not None:
            self._resolved_dispatch_model = selected
        return selected

    def mark_model_unavailable(self, model: str) -> None:
        self.unavailable_models.add(model)
        self.emit({"event": "router_model_unavailable", "model": model})

    def start_attempt(self, *, kind: str, model: str = "", provider: str = "", role: str = "") -> str:
        attempt_id = uuid.uuid4().hex
        self.emit({"event": "attempt_started", "attempt_id": attempt_id,
                   "kind": kind, "model": model, "provider": provider, "role": role,
                   "status": "pending", "cost_usd": None, "cost_known": False})
        return attempt_id

    def guard_request(self, request: Any) -> Any:
        """Enforce hard reservations and capabilities before recording dispatch."""
        if self._accounting_failed:
            raise EpisodeRoutingError("Durable episode accounting previously failed")
        if self._selected and not self._prepared:
            raise EpisodeRoutingError("Prepare the entire configuration before a provider call")
        if self._prepared:
            allowed = ({self._resolved_dispatch_model} if self.decision_resolver is not None
                       else {self._selected.get(role, self._selected.get("*", model))
                             for role, model in self._baselines.items()})
            if request.model not in allowed:
                raise EpisodeRoutingError("Provider dispatch model differs from the frozen episode configuration")
        if self.max_model_calls is not None:
            dispatched = sum(event.get("event") == "attempt_started" and event.get("kind") == "model"
                             for event in self.events)
            if dispatched >= self.max_model_calls:
                raise EpisodeRoutingError("Episode model-call limit exhausted before dispatch")
        metadata = self._metadata.get(request.model, {})
        if metadata.get("forbid_paid_addons"):
            extra = deepcopy(request.extra or {})
            if any(key in extra for key in (
                "plugins", "web_search_options", "web_search", "models", "route", "modalities", "audio", "image_config"
            )):
                raise EpisodeRoutingError("Engineering token-only budget forbids paid add-ons or model overrides")
            provider = extra.get("provider", {})
            if not isinstance(provider, dict):
                raise EpisodeRoutingError("Invalid engineering provider configuration")
            limits = {"prompt": metadata.get("reservation_input_per_1m"),
                      "completion": metadata.get("reservation_output_per_1m"), "request": 0.0, "image": 0.0}
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in limits.values()):
                raise EpisodeRoutingError("Missing engineering provider price bounds")
            previous = provider.get("max_price", {})
            if not isinstance(previous, dict):
                raise EpisodeRoutingError("Invalid provider price cap")
            for key, value in previous.items():
                if key in limits:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise EpisodeRoutingError("Invalid provider price cap")
                    limits[key] = min(limits[key], value)
            extra["provider"] = {**provider, "max_price": limits}
            request = replace(request, extra=extra)
        if metadata.get("provider_only"):
            endpoint = metadata["provider_only"]
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise EpisodeRoutingError("Invalid pinned provider")
            extra = deepcopy(request.extra or {})
            provider = dict(extra.get("provider", {}))
            provider.update({"only": [endpoint], "order": [endpoint], "allow_fallbacks": False})
            extra["provider"] = provider
            request = replace(request, extra=extra)
        required = set(self._required_capabilities)
        input_bound = 16
        for message in request.messages:
            input_bound += 16 + len(message.role.encode())
            if isinstance(message.content, str):
                input_bound += len(message.content.encode())
            else:
                for block in message.content:
                    if block.get("type") in {"image_url", "input_image"}:
                        required.add("vision")
                        if self.max_input_tokens is not None and self.image_input_token_bound is None:
                            raise EpisodeRoutingError("Bounded image requests require an image input-token bound")
                        input_bound += self.image_input_token_bound or 0
                    else:
                        input_bound += len(json.dumps(block, ensure_ascii=False).encode())
            input_bound += len(json.dumps(getattr(message, "tool_calls", [])).encode())
        extra = request.extra or {}
        if extra.get("tools"):
            required.add("tools")
        input_bound += len(json.dumps({k: v for k, v in extra.items()
                                      if k != "openrouter_capability_snapshot"}, ensure_ascii=False).encode())
        if self.max_input_tokens is not None and input_bound > self.max_input_tokens:
            raise EpisodeRoutingError("Episode input-token bound exceeded before dispatch")
        pricing_limit = self._metadata.get(request.model, {}).get("reservation_input_token_limit")
        if pricing_limit is not None and (type(pricing_limit) is not int or input_bound > pricing_limit):
            raise EpisodeRoutingError("Request exceeds frozen pricing-tier input bound")
        metadata = self._metadata.get(request.model, {})
        # Baseline configuration is preserved; changed models must prove support
        # for the actual request, including modalities introduced by tool state.
        selected = set(self._selected.values())
        if request.model in selected or self.decision_resolver is not None:
            if not required.issubset(set(metadata.get("capabilities") or [])):
                raise EpisodeRoutingError("Selected model lacks actual request capabilities")
        output_cap = self.max_output_tokens
        context_window = metadata.get("context_window") or metadata.get("context_length")
        if self.use_full_output_limit and metadata.get("native_output_tokens") is not None:
            native = metadata["native_output_tokens"]
            if type(native) is not int or native <= 0:
                raise EpisodeRoutingError("Invalid frozen native output allowance")
            output_cap = min(native, output_cap) if output_cap is not None else native
            if context_window:
                output_cap = min(output_cap, int(context_window) - input_bound)
            if output_cap <= 0:
                raise EpisodeRoutingError("No output headroom in the model context")
        if output_cap is not None:
            request = replace(request, max_tokens=output_cap if self.use_full_output_limit
                              else min(request.max_tokens or output_cap, output_cap))
        if context_window and input_bound + (request.max_tokens or 0) > int(context_window):
            raise EpisodeRoutingError("Request exceeds the selected model context window")
        if self.decision_resolver is not None:
            self._resolved_dispatch_model = None
        if metadata.get("openrouter_capability_snapshot") is not None:
            extra = deepcopy(request.extra or {})
            extra["openrouter_capability_snapshot"] = deepcopy(metadata["openrouter_capability_snapshot"])
            request = replace(request, extra=extra)
        return request

    def price_response(self, response: Any, fallback_pricing: Any) -> float | None:
        """Use provider billing, then the frozen policy price snapshot if supplied."""
        from theriad.accounting.actuals import (
            provider_reported_cost,
            realized_cost,
            usage_is_available,
        )

        billed = provider_reported_cost(response)
        if billed is not None:
            return billed
        metadata = self._metadata.get(response.model)
        if metadata is None:
            return realized_cost(response, fallback_pricing)
        if metadata.get("require_provider_billing"):
            # Token estimates do not establish settled cost with cache/tier pricing.
            # Explicitly free models retain the existing provable zero bound.
            return zero_charge_failure_bound(response.model, response.provider, metadata, {})
        if not usage_is_available(response):
            return None
        prices = (metadata.get("input_per_1m"), metadata.get("output_per_1m"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in prices):
            return None
        return (response.prompt_tokens * prices[0] + response.completion_tokens * prices[1]) / 1_000_000

    def finish_attempt(self, attempt_id: str, record: dict[str, Any]) -> dict[str, Any]:
        item = self.emit({**record, "event": "attempt_finished", "attempt_id": attempt_id})
        self.attempts.append(item)
        return item

    def record_charge(
        self, kind: str, cost_usd: float | None, *, provider: str = "", role: str = "",
        elapsed_ms: float = 0.0, status: str = "success",
    ) -> dict[str, Any]:
        """Account for a tool, judge, embedding, or router charge, even unknown."""
        if cost_usd is not None and (not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("cost_usd must be a finite nonnegative number or None")
        attempt_id = self.start_attempt(kind=kind, provider=provider, role=role)
        return self.finish_attempt(attempt_id, {
            "kind": kind, "provider": provider, "role": role, "status": status,
            "cost_usd": cost_usd, "cost_known": cost_usd is not None,
            "latency_ms": max(0.0, elapsed_ms),
        })

    def verify_response_backend(self, model: str, backend: str | None) -> None:
        expected = self._metadata.get(model, {}).get("expected_backend")
        if expected is not None and backend != expected:
            self.reject_response("unexpected_backend")
            raise EpisodeRoutingError("Response backend differs from the pinned diagnostic endpoint")

    def reject_response(self, reason: str) -> None:
        """Persist safe validation classifications without duplicating spend."""
        if self.attempts:
            self.emit({"event": "response_rejected", "attempt_id": self.attempts[-1]["attempt_id"],
                       "reason": reason})

    def summary(self) -> dict[str, Any]:
        records = recover_attempts(self.events)
        known = all(item.get("cost_known") is True for item in records)
        known_total = sum(float(item["cost_usd"]) for item in records if item.get("cost_known"))
        return {
            "known_total_cost_usd": known_total,
            "total_cost_usd": known_total if known else None,
            "all_costs_known": known,
            "pending_ids": [item["attempt_id"] for item in records if item.get("status") == "pending"],
            "failed_count": sum(item.get("status") in {"failed", "timeout", "cancelled"} for item in records),
            "attempt_count": len(records),
        }

    async def run_tool(self, role: str, operation: Callable[[], Awaitable[Any]], *, provider: str = "") -> Any:
        """Track a real tool operation; unpriced operations stay unknown."""
        attempt_id = self.start_attempt(kind="tool", provider=provider, role=role)
        started = time.perf_counter()
        try:
            result = await operation()
        except BaseException as exc:
            self.finish_attempt(attempt_id, {
                "kind": "tool", "provider": provider, "role": role, "status": "failed",
                "error_type": type(exc).__name__, "cost_usd": None, "cost_known": False,
                "latency_ms": (time.perf_counter() - started) * 1000,
            })
            raise
        cost = self._tool_prices.get(role)
        self.finish_attempt(attempt_id, {
            "kind": "tool", "provider": provider, "role": role, "status": "success",
            "cost_usd": cost, "cost_known": cost is not None,
            "latency_ms": (time.perf_counter() - started) * 1000,
        })
        return result


def baseline_models_for_provider(models: Mapping[str, tuple[str, str]], provider: str) -> dict[str, str]:
    """Freeze the actual direct-provider tier overrides before policy selection."""
    from theriad.core.providers.runtime import get_provider_model_for_tier
    return {role: model if provider == "theriad" else get_provider_model_for_tier(provider, tier)
            for role, (model, tier) in models.items()}


def recover_attempts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover pending/failed calls as unknown cost, never as zero-cost success."""
    attempts: dict[str, dict[str, Any]] = {}
    for event in events:
        attempt_id = event.get("attempt_id")
        if event.get("event") in {"attempt_started", "attempt_finished"} and attempt_id:
            attempts[attempt_id] = dict(event)
        elif event.get("event") == "response_rejected" and attempt_id in attempts:
            attempts[attempt_id]["response_rejected"] = True
            attempts[attempt_id]["rejection_reason"] = event.get("reason")
    return list(attempts.values())
