"""
Model pricing table for cost estimation.

Design decisions:
- Prices are stored as USD per 1M tokens (input/output separately).
  This is the industry-standard unit used by all providers.
- The live OpenRouter catalog is the source of truth. ``PricingTable`` overlays
  it on top of the built-in seeds below; see ``theriad.accounting.live_prices``
  for the refresh, caching, and failure rules.
- The seeds in ``_DEFAULT_PRICES`` are a *floor*, not a record. They cover slugs
  the catalog does not carry (direct OpenAI/Anthropic/Groq endpoints, local
  models) and keep a fully cold, fully offline start priceable.
- ``PricingTable(use_live=False)`` bypasses the network entirely and resolves
  from the seeds alone. This is genuinely deterministic — the disk cache is not
  merged in at import time — so tests can pin exact values.
- Fallback: if model not found, cost = 0 (we log a warning, never block routing).
  Because that degrades silently to "free", live_prices deliberately serves
  stale prices rather than reporting unknown when the catalog is unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from theriad.core.logging.logger import get_logger

logger = get_logger("accounting.pricing")


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float    # USD per 1M input (prompt) tokens
    output_per_1m: float   # USD per 1M output (completion) tokens


# Seed prices, per 1M tokens (input, output). These are the offline floor; the
# live OpenRouter catalog overrides any slug it carries. Reconciled 2026-07-28.
_DEFAULT_PRICES: dict[str, ModelPrice] = {
    # OpenAI direct (Responses API)
    "gpt-5.4-mini":                    ModelPrice(0.75, 4.50),
    "gpt-5.6-terra":                   ModelPrice(2.50, 15.00),
    "gpt-5.6-sol":                     ModelPrice(5.00, 30.00),
    # Anthropic direct (Messages API)
    "claude-haiku-4-5-20251001":       ModelPrice(1.00, 5.00),
    "claude-sonnet-5":                 ModelPrice(3.00, 15.00),
    "claude-opus-4-8":                 ModelPrice(5.00, 25.00),
    # Groq direct (fast lane)
    "openai/gpt-oss-20b":              ModelPrice(0.075, 0.30),
    "openai/gpt-oss-120b":             ModelPrice(0.15, 0.60),
    # DeepInfra direct
    "deepseek-ai/DeepSeek-V4-Flash":   ModelPrice(0.09, 0.18),
    "deepseek-ai/DeepSeek-V4-Pro":     ModelPrice(1.30, 2.60),
    # OpenAI via OpenRouter
    "openai/gpt-4o":             ModelPrice(2.50, 10.00),
    "openai/gpt-4o-mini":        ModelPrice(0.15, 0.60),
    "openai/gpt-3.5-turbo":      ModelPrice(0.50, 1.50),
    "openai/o1-mini":            ModelPrice(3.00, 12.00),
    # Anthropic via OpenRouter
    "anthropic/claude-3-haiku":  ModelPrice(0.25, 1.25),
    "anthropic/claude-3-sonnet": ModelPrice(3.00, 15.00),
    "anthropic/claude-3-opus":   ModelPrice(15.00, 75.00),
    "anthropic/claude-sonnet-4-5": ModelPrice(3.00, 15.00),
    "anthropic/claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "anthropic/claude-opus-4-5":   ModelPrice(15.00, 75.00),
    # Meta / Mistral (open models via OpenRouter)
    "meta-llama/llama-3-8b-instruct":  ModelPrice(0.06, 0.06),
    "meta-llama/llama-3-70b-instruct": ModelPrice(0.59, 0.79),
    "mistralai/mistral-7b-instruct":   ModelPrice(0.07, 0.07),
    # Qwen3 family (cheap_code / code_strong tiers)
    "qwen/qwen3-8b":          ModelPrice(0.06, 0.06),
    "qwen/qwen3-14b":         ModelPrice(0.10, 0.10),
    "qwen/qwen3-30b-a3b":     ModelPrice(0.10, 0.10),
    "qwen/qwen3-235b-a22b":   ModelPrice(0.14, 0.14),
    # DeepSeek (cheap_code / code_strong / strong_reasoning / premium)
    "deepseek/deepseek-r1-0528":  ModelPrice(0.14, 0.28),
    "deepseek/deepseek-v4-flash": ModelPrice(0.14, 0.28),
    # V4 Flash 0731: identical architecture and list price to the April release
    # above, but +10 on the AA Intelligence Index (40 -> 50). Seeded at DeepSeek's
    # direct rate; OpenRouter currently serves it for 0.09/0.18, and as with Luna
    # the cheaper rate is documented rather than baked in so the eval budget
    # guard errs toward over-estimating spend.
    "deepseek/deepseek-v4-flash-0731": ModelPrice(0.14, 0.28),
    # Google (strong_reasoning)
    "google/gemini-2.5-flash":    ModelPrice(0.15, 0.60),
    # Moonshot Kimi (legacy explicit-request compatibility; no longer active
    # in the default registry). OpenRouter may return a versioned slug like
    # moonshotai/kimi-k2.7-code-20260612 in the response.
    # Both are added explicitly; the prefix-match fallback in PricingTable.get()
    # also handles new versioned releases automatically.
    "kimi/kimi-k2":                         ModelPrice(0.60, 2.50),
    "moonshotai/kimi-k2.7-code":            ModelPrice(0.73, 3.50),
    "moonshotai/kimi-k2.7-code-20260612":   ModelPrice(0.73, 3.50),
    # Qwen 3.6 Plus (cheap_code / code_strong fallback)
    "qwen/qwen3.6-plus":                    ModelPrice(0.325, 1.95),
    # Xiaomi MiMo (cheap_bulk / free_or_floor)
    "xiaomi/mimo-vl-7b-rl":           ModelPrice(0.08, 0.08),
    "xiaomi/mimo-vl-7b-rl:free":      ModelPrice(0.0, 0.0),
    # Xiaomi MiMo v2.5 — OpenRouter may return a versioned slug like
    # xiaomi/mimo-v2.5-20260422. Both the canonical and versioned slugs are
    # listed explicitly; the prefix-match fallback covers future versioned releases.
    "xiaomi/mimo-v2.5":               ModelPrice(0.14, 0.28),
    "xiaomi/mimo-v2.5-20260422":      ModelPrice(0.14, 0.28),
    # Z-AI GLM 5.2 (premium tier)
    "z-ai/glm-5.2":                              ModelPrice(0.7686, 2.4156),
    # ── Premium/frontier candidates (see docs/non-chinese-model-costs.md)
    # Seeded at LIST price, not the promotional OpenRouter rate. Seeds feed the
    # budget guard in theriad.eval, and over-estimating spend is the safe error;
    # the live catalog overlay picks up the real rate at runtime anyway.
    # GPT-5.6 Luna — OpenAI list. OpenRouter currently discounts to 0.10/0.60,
    # which would make this the cheapest premium candidate by a wide margin.
    # Do not bake the promo into the seed: promos lapse, and a seed that under-
    # states cost is the one failure mode the budget guard cannot catch.
    "openai/gpt-5.6-luna":                       ModelPrice(0.20, 1.20),
    # GPT-5.6 Sol — OpenAI list rate used by the public landing comparison.
    "openai/gpt-5.6-sol":                        ModelPrice(5.00, 30.00),
    # Grok 4.5 (xAI, US). Cached input is 0.50/1M, a 75% discount.
    "x-ai/grok-4.5":                             ModelPrice(2.00, 6.00),
    # Same rate as 4.5, which it supersedes. NB: xAI doubles to 4.00/12.00 for
    # requests over 200K tokens and applies that rate to *every* token in the
    # request, so this seed understates long-context calls by 2x.
    "x-ai/grok-4.6":                             ModelPrice(2.00, 6.00),
    # Claude Opus 5 (Anthropic, US). Bedrock regional endpoints carry a ~10%
    # premium over global; this seed is the OpenRouter list rate.
    "anthropic/claude-opus-5":                   ModelPrice(5.00, 25.00),
    # GLM 5.3 Flash (Z.AI, CN). OpenRouter currently advertises a temporary 50%
    # discount at $0.075/$0.25; keep the non-promotional list seed here so a
    # discount expiry cannot make offline cost estimates optimistic. The live
    # catalog overlay picks up the discounted rate while it is available.
    "z-ai/glm-5.3-flash":                         ModelPrice(0.15, 0.50),
    # Muse Spark 1.2 (Meta, US). Live OpenRouter catalog rate; cached input is
    # 0.15/1M. This remains the U.S.-only frontier fallback.
    "meta/muse-spark-1.2":                       ModelPrice(1.25, 4.25),
    # Same-family successors adopted 2026-09-24 (live OpenRouter list prices).
    "x-ai/grok-4.7":                             ModelPrice(1.60, 4.80),
    "meta/muse-spark-1.3":                       ModelPrice(1.25, 4.25),
    "openai/gpt-6-luna":                         ModelPrice(0.10, 0.50),
    "xiaomi/mimo-v2.6-flash":                    ModelPrice(0.14, 0.28),
    "qwen/qwen3.7-plus":                         ModelPrice(0.32, 1.28),
    "deepseek/deepseek-v4.1-flash":              ModelPrice(0.14, 0.42),
    "z-ai/glm-5.3":                              ModelPrice(0.84, 2.64),
    # Muse Spark 1.1 remains priceable as a legacy premium fallback.
    "meta/muse-spark-1.1":                       ModelPrice(1.25, 4.25),
    # Nvidia Nemotron free
    "nvidia/llama-3.1-nemotron-ultra-253b:free": ModelPrice(0.0, 0.0),
    "nvidia/nemotron-3-super-120b-a12b:free":    ModelPrice(0.0, 0.0),
    "nvidia/nemotron-3-nano-30b-a3b:free":       ModelPrice(0.0, 0.0),
    # Cohere North Mini Code free (benchmark-only; candidate for free_or_floor_code / cheap_code)
    # pricing_source: openrouter_free — zero cost, subject to rate-limit constraints.
    "cohere/north-mini-code:free":               ModelPrice(0.0, 0.0),
    # Ollama (local = $0)
    "llama3":     ModelPrice(0.0, 0.0),
    "mistral":    ModelPrice(0.0, 0.0),
    "gemma":      ModelPrice(0.0, 0.0),
    "phi3":       ModelPrice(0.0, 0.0),
}


class PricingLookup(NamedTuple):
    """Result of a PricingTable.lookup() call."""
    price: ModelPrice
    source: str       # "exact" | "alias"
    model_slug: str   # the table key that was matched


class PricingTable:
    """Lookup table for model pricing with fallback.

    Resolution order, cheapest-to-most-authoritative:
      1. ``_DEFAULT_PRICES`` seeds (always present, offline-safe)
      2. the live OpenRouter catalog, when ``use_live`` is True
      3. explicit ``prices`` passed by the caller (always wins)
    """

    def __init__(
        self,
        prices: dict[str, ModelPrice] | None = None,
        *,
        use_live: bool = True,
    ):
        self._prices = dict(_DEFAULT_PRICES)
        self._live_stale: bool | None = None
        if use_live:
            live_prices = self._live_overlay()
            self._prices.update(live_prices)
            # Seeded versioned aliases must follow the live canonical model
            # price. Otherwise a refreshed catalog can update
            # ``moonshotai/kimi-k2.7-code`` while the seeded dated alias keeps
            # an older value and produces inconsistent realized costs.
            for canonical, live_price in live_prices.items():
                for alias in _DEFAULT_PRICES:
                    suffix = alias[len(canonical) + 1:] if alias.startswith(f"{canonical}-") else ""
                    if suffix.isdigit() and len(suffix) >= 6:
                        self._prices[alias] = live_price
        if prices:
            self._prices.update(prices)

    @staticmethod
    def _live_overlay_snapshot():
        from theriad.accounting.live_prices import get_live_snapshot

        return get_live_snapshot()

    def _live_overlay(self) -> dict[str, ModelPrice]:
        """Live catalog prices, or an empty overlay if nothing is resolvable.

        Never raises: a pricing failure must not take routing down with it.
        """
        try:
            snapshot = self._live_overlay_snapshot()
        except Exception as exc:                     # pragma: no cover - defensive
            logger.warning("live price overlay unavailable", extra={"error": str(exc)})
            return {}
        self._live_stale = snapshot.stale
        return {
            slug: ModelPrice(pin, pout)
            for slug, (pin, pout) in snapshot.prices.items()
        }

    @property
    def live_stale(self) -> bool | None:
        """True/False once a live overlay was attempted; None when it was not."""
        return self._live_stale

    def lookup(self, model: str) -> PricingLookup | None:
        """
        Return PricingLookup(price, source, model_slug) or None if not found.

        Lookup priority:
          1. Exact match          → source="exact"
          2. Prefix/suffix match  → source="alias"  (versioned slugs, bare names)
          3. Not found            → None
        """
        # Exact match
        if model in self._prices:
            return PricingLookup(self._prices[model], "exact", model)
        # Prefix match: versioned slugs like "moonshotai/kimi-k2.7-code-20260612"
        # Suffix match: bare slug like "gpt-4o-mini" → "openai/gpt-4o-mini"
        for key in self._prices:
            if (
                model.startswith(key)                      # versioned slug prefix
                or key.endswith(model)                     # bare model name
                or model.endswith(key.split("/")[-1])      # basename match
            ):
                return PricingLookup(self._prices[key], "alias", key)
        logger.debug("model price unknown", extra={"model": model})
        return None

    def get(self, model: str) -> ModelPrice | None:
        """Return ModelPrice or None. Backward-compatible wrapper around lookup()."""
        result = self.lookup(model)
        return result.price if result else None

    def update(self, model: str, price: ModelPrice) -> None:
        self._prices[model] = price


# ── Platform fee ──────────────────────────────────────────────────────────────

PLATFORM_FEE_PER_1M_TOKENS: float = 0.001
"""Illustrative routed-cost fee, in USD per 1M tokens.

0.001 USD per 1M tokens is one tenth of a cent per million tokens. The billing
boundary applies it only to successful authenticated paid calls; guest and
free-demo traffic is exempt. Commercial terms are defined in
``theriad.pricing_plans``.
"""


def platform_fee(prompt_tokens: int, completion_tokens: int) -> float:
    """Return Theriad's platform fee in USD for the given token count.

    This arithmetic helper is neutral about caller identity. The billing
    service applies it only to successful authenticated paid calls; guest and
    free-demo traffic remains uncharged.
    """
    total_tokens = max(prompt_tokens, 0) + max(completion_tokens, 0)
    return total_tokens * PLATFORM_FEE_PER_1M_TOKENS / 1_000_000


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    table: PricingTable | None = None,
    *,
    include_platform_fee: bool = False,
) -> float:
    """Return estimated USD cost. Model cost is 0.0 if model is not in table.

    When ``include_platform_fee`` is True, the illustrative routed-cost fee
    (see :func:`platform_fee`) is added on top, so the estimate can be non-zero
    even for free or unknown models. Defaults to False so pure model-cost
    callers are unaffected.
    """
    t = table or PricingTable()
    price = t.get(model)
    model_cost = 0.0 if price is None else (
        prompt_tokens * price.input_per_1m + completion_tokens * price.output_per_1m
    ) / 1_000_000
    if include_platform_fee:
        model_cost += platform_fee(prompt_tokens, completion_tokens)
    return model_cost
