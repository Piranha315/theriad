"""
Which models can actually see an image.

Design rules:

1. **The live catalog decides, not the slug name.** OpenRouter reports
   ``architecture.input_modalities`` per model; ``live_prices`` captures it. A
   name-based guess is worse than useless here — routing an image to a
   text-only model produces a confident answer about nothing, which is harder
   to notice than an outright error.

2. **Seeds are a floor, not a record**, exactly as with prices. They exist so a
   cold, offline start can still answer "can this model see?" for the handful of
   slugs the registry actually routes to.

3. **Unknown means no.** If neither the catalog nor the seeds vouch for a slug,
   it is treated as text-only. The cost of a false negative is falling back to a
   known-good vision model; the cost of a false positive is a silently wrong
   answer. Those are not symmetric.

The default vision model is ``xiaomi/mimo-v2.6-flash`` (successor to
``mimo-v2.5``, adopted 2026-09-24): natively omnimodal, already an active
OpenRouter registry candidate, and therefore needs no new provider, dependency,
or tier. Escalation uses the current premium comparison candidate
``openai/gpt-6-luna``.
"""
from __future__ import annotations

from theriad.core.logging.logger import get_logger

logger = get_logger("models.vision")

DEFAULT_VISION_MODEL = "xiaomi/mimo-v2.6-flash"
"""Preset model that interprets attached images unless something overrides it."""

VISION_ESCALATION_MODEL = "openai/gpt-6-luna"
"""Stronger fallback when the default cannot serve the request."""

SEED_VISION_MODELS: frozenset[str] = frozenset({
    # Xiaomi MiMo — natively omnimodal.
    "xiaomi/mimo-v2.6-flash",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-20260422",
    "xiaomi/mimo-vl-7b-rl",
    "xiaomi/mimo-vl-7b-rl:free",
    # Current registry candidate plus legacy direct-provider/catalog aliases.
    # The aliases remain accepted for compatibility; they are not Theriad's
    # default cross-provider route.
    "openai/gpt-6-luna",
    "openai/gpt-5.6-luna",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    # Muse Spark 1.2 and 1.3 are multimodal in the live OpenRouter catalog.
    "meta/muse-spark-1.3",
    "meta/muse-spark-1.2",
    # Anthropic — all Claude 3+ models accept images.
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "anthropic/claude-opus-5",
    # Google.
    "google/gemini-2.5-flash",
    # Z-AI GLM.
    "z-ai/glm-5.2",
    "z-ai/glm-5.3-flash",
})
"""Offline floor. Deliberately conservative — omission costs a fallback, a wrong
inclusion costs a confidently wrong answer about an image the model never saw."""


def _live_vision_slugs() -> frozenset[str]:
    """Vision slugs per the live catalog, or empty if it is unavailable."""
    try:
        from theriad.accounting.live_prices import get_live_snapshot

        return get_live_snapshot().vision
    except Exception as exc:                         # pragma: no cover - defensive
        logger.warning("live vision lookup failed", extra={"error": str(exc)})
        return frozenset()


def supports_vision(model: str, *, use_live: bool = True) -> bool:
    """True when ``model`` can accept image input.

    Checks the live catalog first, then the seed floor. Versioned slugs
    (``xiaomi/mimo-v2.5-20260422``) fall back to a prefix match against their
    base slug, mirroring how PricingTable resolves prices.
    """
    if not model:
        return False
    if use_live:
        live = _live_vision_slugs()
        if model in live:
            return True
        # Only trust a prefix match against the live set when the live set is
        # actually populated; an empty set means "we don't know", not "no".
        if live and any(model.startswith(f"{slug}-") for slug in live):
            return True
    if model in SEED_VISION_MODELS:
        return True
    return any(model.startswith(f"{slug}-") for slug in SEED_VISION_MODELS)


def vision_candidates(
    candidates: list[tuple[str, str]], *, use_live: bool = True
) -> list[tuple[str, str]]:
    """Keep only ``(provider, model)`` pairs that can see an image.

    Order is preserved, so tier preference survives the filter.
    """
    return [
        (provider, model)
        for provider, model in candidates
        if supports_vision(model, use_live=use_live)
    ]


def ensure_vision_candidates(
    candidates: list[tuple[str, str]],
    *,
    provider: str = "openrouter",
    use_live: bool = True,
) -> list[tuple[str, str]]:
    """Return a candidate list guaranteed to be able to see an image.

    Filters to vision-capable candidates and, if that empties the list (or the
    escalation model is missing), appends the presets. A request carrying an
    image must never be handed to a text-only model, so this always returns at
    least the default vision model.
    """
    kept = vision_candidates(candidates, use_live=use_live)
    existing = {model for _, model in kept}
    for preset in (DEFAULT_VISION_MODEL, VISION_ESCALATION_MODEL):
        if preset not in existing:
            kept.append((provider, preset))
            existing.add(preset)
    return kept
