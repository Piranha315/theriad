"""
Executors: adapters that run one eval item against one model.

Kept separate from ``run`` so the harness core imports no provider code and the
offline paths (plan, report) never touch the network.

An executor returns a ``CallResult``. The ``truncated`` flag is the important
one: a response cut off by an output-token cap is missing data, not a wrong
answer, and the grid excludes it from the success-rate estimate rather than
scoring it as a failure.

Output caps and measurement validity
------------------------------------
Two different numbers get confused here, and conflating them silently corrupts
the grid:

  - the *assumed* output length, used to estimate cost before spending;
  - the *generation cap* sent to the provider as max_tokens.

They are not the same number. A 400-token assumption is fine for costing a
short-answer corpus, but imposing it as max_tokens truncates any model that
reasons before answering — and it truncates hardest on the difficult items where
reasoning is longest. The measured result is then "this model is bad at hard
questions" when the truth is "this model was cut off mid-sentence".

So the generation cap defaults to None (no cap sent at all) and is separate from
the costing assumption. When a cap is set, truncation is detected and excluded.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from .corpus import EvalItem


@dataclass(frozen=True)
class CallResult:
    """One executed call."""
    output: str
    cost_usd: float = 0.0
    truncated: bool = False
    finish_reason: str | None = None


# Finish reasons that mean the provider stopped for length, not completion.
_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})


def is_truncated(finish_reason: str | None) -> bool:
    """True when a finish reason indicates an output-cap cutoff."""
    return finish_reason is not None and finish_reason in _TRUNCATION_REASONS


# ── OpenRouter provider preferences ───────────────────────────────────────────
#
# OpenRouter serves most models through several upstream providers at differing
# price, throughput and reliability. Left unset it picks for you, which is fine
# in production and bad in a benchmark: a grid whose cells were served by
# different upstreams is measuring the upstreams as much as the models, and it
# is not reproducible.
#
# theriad/core/providers/openrouter.py already forwards a "provider" key from
# CompletionRequest.extra, so pinning needs no provider changes.

AWS_HIGHEST_THROUGHPUT: dict = {
    "only": ["amazon-bedrock"],
    "sort": "throughput",
    "allow_fallbacks": False,
}
"""Pin to Amazon Bedrock, highest-throughput endpoint first.

``allow_fallbacks: False`` is the load-bearing part. Without it OpenRouter
silently serves from a different upstream when Bedrock is busy, and the grid
records that result as if it came from the pinned one — turning a capacity blip
into a permanent, invisible measurement error.

Note that Bedrock regional endpoints carry roughly a 10% premium over global, so
observed cost may exceed the seeded price. Cost is recorded per call from the
response, so the grid reflects what was actually charged.
"""

META_ONLY: dict = {
    "only": ["meta"],
    "allow_fallbacks": False,
}
"""Pin Muse Spark 1.2 to Meta's OpenRouter provider for reproducible evals."""

DEFAULT_PROVIDER_PREFERENCES: dict[str, dict] = {
    "anthropic/claude-opus-5": AWS_HIGHEST_THROUGHPUT,
    "meta/muse-spark-1.2": META_ONLY,
}
"""Per-model OpenRouter provider pins used when populating the grid."""


RETRY_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
"""HTTP statuses worth retrying: throttling and transient upstream failures.

429 is the one that matters in practice. A sustained sequential run accumulates
rate-limit pressure, and once throttled *every* subsequent request is rejected
instantly — so an un-retried run burns through a whole model's plan in about a
second and records it as total failure.
"""

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_BASE_S = 2.0


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt: honour Retry-After, else backoff."""
    for attr in ("retry_after", "headers"):
        value = getattr(exc, attr, None)
        if isinstance(value, dict):
            value = value.get("retry-after") or value.get("Retry-After")
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    return DEFAULT_BACKOFF_BASE_S * (2 ** (attempt - 1))


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def openrouter_executor(
    generation_max_tokens: int | None = None,
    provider_preferences: dict[str, dict] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Build an executor backed by the OpenRouter provider.

    Args:
        generation_max_tokens: Cap sent to the provider. **None means no cap**,
            which is the default and the right choice for measurement: let the
            model finish. Set an integer only to bound a runaway, and expect the
            truncation counters to tell you if it bit.
        provider_preferences: {model: OpenRouter provider-routing block}. Pins a
            model to a specific upstream so grid cells are reproducible.
            Defaults to DEFAULT_PROVIDER_PREFERENCES.

    Imports the provider lazily so importing this module — or running the
    offline subcommands — never requires credentials.
    """
    prefs = (
        DEFAULT_PROVIDER_PREFERENCES
        if provider_preferences is None
        else provider_preferences
    )

    def _execute(model: str, item: EvalItem) -> CallResult:
        from theriad.accounting.pricing import estimate_cost
        from theriad.core.providers.interfaces import CompletionRequest, Message
        from theriad.core.providers.openrouter import OpenRouterProvider

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; cannot populate the grid."
            )
        provider = OpenRouterProvider(api_key=api_key)
        # item.max_output_tokens is a per-item override and stays None unless the
        # corpus author deliberately bounded that item.
        cap = item.max_output_tokens or generation_max_tokens
        extra: dict = {}
        if model in prefs:
            extra["provider"] = prefs[model]
        request = CompletionRequest(
            # CompletionRequest.messages is list[Message], not list[dict]; the
            # provider reads message.role / message.content as attributes.
            messages=[Message(role="user", content=item.prompt)],
            model=model,
            extra=extra,
            **({"max_tokens": cap} if cap is not None else {}),
        )
        # OpenRouterProvider.complete is async and there is no sync variant.
        # asyncio.run is correct here because populate_grid is sequential and
        # owns the thread; it must not be called from inside a running loop.
        #
        # Retry throttling and transient upstream errors. Without this a 429
        # storm silently converts a model's entire plan into recorded failures.
        last: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = asyncio.run(provider.complete(request))
                break
            except Exception as exc:  # noqa: BLE001 - re-raised below
                status = _status_of(exc)
                retryable = status in RETRY_STATUS_CODES if status is not None else False
                if not retryable or attempt == max_attempts:
                    raise
                last = exc
                time.sleep(_retry_after_seconds(exc, attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise last if last else RuntimeError("retry loop exhausted")
        finish_reason = getattr(response, "finish_reason", None)
        # CompletionResponse carries no cost field, so cost is derived from the
        # reported token counts and the pricing table. Unpriced models resolve
        # to 0.0, which is PricingTable's documented behaviour and matches how
        # `plan` and `report` already treat them.
        cost_usd = estimate_cost(
            model,
            getattr(response, "prompt_tokens", 0) or 0,
            getattr(response, "completion_tokens", 0) or 0,
        )
        return CallResult(
            output=response.content or "",
            cost_usd=cost_usd,
            truncated=is_truncated(finish_reason),
            finish_reason=finish_reason,
        )

    return _execute


def static_executor(
    responses: dict[tuple[str, str], str],
    cost_per_call: float = 0.0,
    truncate: set[tuple[str, str]] | None = None,
):
    """Executor that replays canned outputs. For tests and dry runs.

    ``responses`` maps (model, item_id) → output text; unmapped pairs return an
    empty string, which grades as incorrect. ``truncate`` marks pairs that should
    come back flagged as cut off.
    """
    truncated_pairs = truncate or set()

    def _execute(model: str, item: EvalItem) -> CallResult:
        key = (model, item.id)
        return CallResult(
            output=responses.get(key, ""),
            cost_usd=cost_per_call,
            truncated=key in truncated_pairs,
            finish_reason="length" if key in truncated_pairs else "stop",
        )

    return _execute
