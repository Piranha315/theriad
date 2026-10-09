"""
OpenRouter benchmark lane.

Uses the full theriad routing stack to select the model for each prompt,
then calls OpenRouter via the existing OpenRouterProvider. This gives an
accurate picture of what the live system would do.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from theriad.benchmark.providers.base import BenchmarkLane
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult, _make_output_preview
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.openrouter")

# Models registered for OpenRouter benchmarking — one representative per tier.
# The lane also runs whatever the theriad router selects dynamically.
_OPENROUTER_MODELS = [
    "openrouter/auto",                  # sentinel: let theriad routing decide per-prompt
    "openrouter/auto-or",               # benchmark OpenRouter's native auto-router (calls OR with model="openrouter/auto")
    "cohere/north-mini-code:free",      # benchmark-only: free code candidate (not in production tiers)
]

# Benchmark-only metadata for models not yet in the production registry.
# These models are being evaluated before any routing assignment.
# Keys match slugs in _OPENROUTER_MODELS (excluding openrouter/auto).
_OPENROUTER_BENCHMARK_METADATA: dict[str, dict] = {
    "openrouter/auto-or": {
        "context_window": None,
        "max_output_tokens": None,
        "input_per_million": None,
        "output_per_million": None,
        "pricing_source": "dynamic",
        "strengths": ["automatic_model_selection", "quality_routing"],
        "weaknesses": ["variable_cost", "no_cost_predictability", "latency_variance"],
        "candidate_tiers": [],
        "status": "benchmark_only",
        "notes": "Calls OpenRouter's native auto-router. "
                 "response.model captures which model OR actually selected. "
                 "NOT the default Theriad router — benchmark comparison only.",
    },
    "cohere/north-mini-code:free": {
        "context_window": 256_000,
        "max_output_tokens": 64_000,
        "input_per_million": 0,
        "output_per_million": 0,
        "pricing_source": "openrouter_free",
        "strengths": ["code", "agentic_coding", "terminal_tasks", "tool_use", "free_hosted"],
        "weaknesses": ["unknown_quality", "free_tier_rate_limits"],
        "candidate_tiers": ["free_or_floor_code", "cheap_code"],
        "status": "benchmark_only",
        "notes": "Evaluate before routing production traffic. Compare against llama3.1:8b, "
                 "xiaomi/mimo-v2.5-20260422, and qwen/qwen3.6-plus.",
    },
}


class OpenRouterLane(BenchmarkLane):
    """
    Benchmark lane that routes each prompt through the theriad stack and
    calls OpenRouter. Reports the dynamically selected model in results.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def models(self) -> list[str]:
        if not self._api_key:
            logger.warning("OPENROUTER_API_KEY not set — openrouter lane disabled")
            return []
        return _OPENROUTER_MODELS

    async def run_one(self, model: str, benchmark_prompt: BenchmarkPrompt) -> BenchmarkResult:
        ts = datetime.now(timezone.utc).isoformat()

        if not self._api_key:
            return self._skipped(ts, model, benchmark_prompt, "No OPENROUTER_API_KEY")

        try:
            from theriad.accounting.pricing import PricingTable
            from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderError
            from theriad.core.providers.openrouter import OpenRouterProvider
            from theriad.models.tiers import TIER_TIMEOUT_SECONDS

            if model == "openrouter/auto":
                # ── Routed path: let theriad routing select the model ──────────
                from theriad.layers.layer0.layer import build_default_layer
                from theriad.layers.layer1_features.layer import FeatureRoutingLayer
                from theriad.models.policy import PolicyResolver
                from theriad.models.tiers import TIER_MAX_TOKENS

                layer0 = build_default_layer()
                layer1 = FeatureRoutingLayer()
                ctx: dict = {}
                routing = layer0.route(benchmark_prompt.prompt, ctx) or layer1.route(benchmark_prompt.prompt, ctx)

                if routing is None:
                    from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
                    routing = RoutingResult(
                        decision=RoutingDecision(
                            tier="cheap_bulk", confidence=0.0,
                            reason="all layers abstained",
                        ),
                        layer_name="fallback",
                    )

                tier = routing.decision.tier
                resolution = PolicyResolver().resolve(tier, "balanced", routing.decision.confidence)
                resolved_model = resolution.model

                tier_max = TIER_MAX_TOKENS.get(tier)
                timeout = TIER_TIMEOUT_SECONDS.get(tier, 60.0)
                # Use tier_max as the sole cap — benchmark_prompt.max_output_tokens
                # may be stale (baked into old JSONL files) and should not override
                # the tier system's output budget.
                max_tokens = tier_max  # None when tier has no defined limit

            elif model == "openrouter/auto-or":
                # ── OR native auto-router path ────────────────────────────────
                # Call OpenRouter with model="openrouter/auto" literally so OR's
                # own auto-router selects the model. response.model captures what
                # OR actually chose — that's the key benchmark signal.
                # openrouter/auto-or is NOT the Theriad default; benchmark only.
                resolved_model = "openrouter/auto"
                tier = "benchmark_direct"
                timeout = TIER_TIMEOUT_SECONDS.get("cheap_code", 60.0)
                max_tokens = None  # let OR decide; no prompt-level cap

            else:
                # ── Direct path: benchmark a specific model slug without routing ──
                # Used for benchmark-only candidates not yet in production tiers.
                resolved_model = model
                tier = "benchmark_direct"
                timeout = TIER_TIMEOUT_SECONDS.get("cheap_code", 60.0)
                max_tokens = None  # no tier cap; let the model's own limit apply

            # ── Call provider ─────────────────────────────────────────────────
            provider = OpenRouterProvider(api_key=self._api_key, timeout=timeout)
            req = CompletionRequest(
                model=resolved_model,
                messages=[Message(role="user", content=benchmark_prompt.prompt)],
                max_tokens=max_tokens,
            )

            t0 = time.monotonic()
            try:
                response = await provider.complete(req)
                total_ms = (time.monotonic() - t0) * 1000

                pt = PricingTable()
                pricing_lookup = pt.lookup(response.model)
                actual_cost = None
                cost_known = False
                pricing_source: str | None = None
                pricing_model_slug: str | None = None
                if pricing_lookup is not None:
                    actual_cost = (
                        response.prompt_tokens * pricing_lookup.price.input_per_1m
                        + response.completion_tokens * pricing_lookup.price.output_per_1m
                    ) / 1_000_000
                    cost_known = True
                    pricing_source = pricing_lookup.source
                    pricing_model_slug = pricing_lookup.model_slug
                else:
                    pricing_source = "missing"

                tps = (
                    response.completion_tokens / (response.latency_ms / 1000)
                    if response.latency_ms > 0 and response.completion_tokens > 0 else None
                )

                preview, sha256 = (
                    _make_output_preview(response.content)
                    if response.content else ("", "")
                )

                return BenchmarkResult(
                    timestamp=ts,
                    provider=self.provider_name,
                    model=response.model,          # actual model returned by provider
                    tier=tier,
                    prompt_id=benchmark_prompt.id,
                    category=benchmark_prompt.category,
                    input_tokens=response.prompt_tokens,
                    output_tokens=response.completion_tokens,
                    actual_cost_usd=actual_cost,
                    provider_latency_ms=response.latency_ms,
                    total_latency_ms=total_ms,
                    tokens_per_second=tps,
                    finish_reason=response.finish_reason,
                    status="success",
                    error=None,
                    quality_pass=benchmark_prompt.quality_check(response.content),
                    cost_known=cost_known,
                    pricing_source=pricing_source,
                    pricing_model_slug=pricing_model_slug,
                    requested_model_slug=resolved_model,
                    output_preview=preview or None,
                    output_sha256=sha256 or None,
                )

            except ProviderError as exc:
                total_ms = (time.monotonic() - t0) * 1000
                return self._error(ts, resolved_model, tier, benchmark_prompt,
                                   total_ms, str(exc))
            except Exception as exc:
                total_ms = (time.monotonic() - t0) * 1000
                return self._error(ts, resolved_model, tier, benchmark_prompt,
                                   total_ms, str(exc))

        except Exception as exc:
            return self._error(ts, model, None, benchmark_prompt, 0.0, str(exc))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _error(
        self, ts: str, model: str, tier: str | None, bp: BenchmarkPrompt,
        total_ms: float, error: str, status: str = "error",
    ) -> BenchmarkResult:
        return BenchmarkResult(
            timestamp=ts, provider=self.provider_name, model=model, tier=tier,
            prompt_id=bp.id, category=bp.category,
            input_tokens=0, output_tokens=0, actual_cost_usd=None,
            provider_latency_ms=total_ms, total_latency_ms=total_ms,
            tokens_per_second=None, finish_reason=None,
            status=status, error=error, quality_pass=None,
        )

    def _skipped(self, ts: str, model: str, bp: BenchmarkPrompt, reason: str) -> BenchmarkResult:
        return self._error(ts, model, None, bp, 0.0, reason, status="skipped")
