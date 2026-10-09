"""Tests for ModelRegistry."""
from __future__ import annotations

from theriad.models.registry import DEFAULT_REGISTRY, ModelRegistry
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_FREE_OR_FLOOR,
    TIER_FREE_MOE_TIERS,
    TIER_PREMIUM,
    TIER_STRONG_REASONING,
    ModelCandidate,
)


def test_default_registry_has_all_tiers():
    for tier in [TIER_FREE_OR_FLOOR, TIER_CHEAP_BULK, TIER_CHEAP_CODE,
                 TIER_CODE_STRONG, TIER_STRONG_REASONING, TIER_PREMIUM]:
        candidates = DEFAULT_REGISTRY.candidates_for_tier(tier)
        assert len(candidates) >= 1, f"tier '{tier}' has no candidates"

    for tier in TIER_FREE_MOE_TIERS:
        candidates = DEFAULT_REGISTRY.candidates_for_tier(tier)
        assert len(candidates) >= 2, f"MoE tier '{tier}' needs a fallback"


def test_every_tier_has_primary():
    for tier in DEFAULT_REGISTRY.all_tiers():
        primary = DEFAULT_REGISTRY.primary_for_tier(tier)
        assert primary is not None, f"tier '{tier}' has no primary candidate"
        assert primary.model, "model slug must be non-empty"


def test_candidates_have_required_fields():
    for tier in DEFAULT_REGISTRY.all_tiers():
        for c in DEFAULT_REGISTRY.candidates_for_tier(tier):
            assert c.tier == tier
            assert c.provider in {"openrouter", "ollama"}
            assert c.model
            assert c.role in {"primary", "fallback", "premium_fallback", "experimental"}


def test_unknown_tier_returns_empty():
    candidates = DEFAULT_REGISTRY.candidates_for_tier("nonexistent_tier")
    assert candidates == []


def test_primary_for_unknown_tier_returns_none():
    assert DEFAULT_REGISTRY.primary_for_tier("nonexistent") is None


def test_fallback_chain_excludes_primary():
    for tier in DEFAULT_REGISTRY.all_tiers():
        primary = DEFAULT_REGISTRY.primary_for_tier(tier)
        fallbacks = DEFAULT_REGISTRY.fallback_chain_for_tier(tier)
        if primary:
            assert primary not in fallbacks


def test_custom_registry():
    custom_candidate = ModelCandidate(
        tier=TIER_CHEAP_BULK,
        provider="openrouter",
        model="custom/test-model",
        role="primary",
        input_cost_per_million=0.01,
        output_cost_per_million=0.01,
    )
    registry = ModelRegistry({TIER_CHEAP_BULK: [custom_candidate]})
    assert registry.primary_for_tier(TIER_CHEAP_BULK) == custom_candidate
    assert registry.candidates_for_tier(TIER_PREMIUM) == []


def test_candidates_are_list_copies():
    """Returned lists must be copies, not references (mutation safety)."""
    c1 = DEFAULT_REGISTRY.candidates_for_tier(TIER_CHEAP_BULK)
    c2 = DEFAULT_REGISTRY.candidates_for_tier(TIER_CHEAP_BULK)
    assert c1 is not c2


def test_openai_is_fallback_not_primary_in_cheap_tiers():
    """ADR D-019: OpenAI must NOT be the primary model in cheap tiers."""
    for tier in [TIER_CHEAP_BULK, TIER_CHEAP_CODE]:
        primary = DEFAULT_REGISTRY.primary_for_tier(tier)
        assert primary is not None
        assert not primary.model.startswith("openai/"), (
            f"OpenAI should not be primary in '{tier}': got {primary.model}"
        )
