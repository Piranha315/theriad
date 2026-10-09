"""Tests for tier constants and ModelCandidate."""
from __future__ import annotations

from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_FREE_OR_FLOOR,
    TIER_FREE_MOE_TIERS,
    TIER_FRONTIER,
    TIER_PREMIUM,
    TIER_STRONG_REASONING,
    VALID_TIERS,
    ModelCandidate,
)


def test_all_tiers_in_valid_set():
    expected = {
        TIER_FREE_OR_FLOOR, TIER_CHEAP_BULK, TIER_CHEAP_CODE,
        TIER_CODE_STRONG, TIER_STRONG_REASONING, TIER_PREMIUM,
        TIER_FRONTIER,
        *TIER_FREE_MOE_TIERS,
    }
    assert expected == VALID_TIERS


def test_tier_strings_are_snake_case():
    for tier in VALID_TIERS:
        assert tier == tier.lower()
        assert " " not in tier


def test_model_candidate_defaults():
    c = ModelCandidate(tier=TIER_CHEAP_BULK, provider="openrouter", model="test/model")
    assert c.role == "primary"
    assert c.input_cost_per_million is None
    assert c.output_cost_per_million is None


def test_model_candidate_cost_estimate():
    c = ModelCandidate(
        tier=TIER_CHEAP_BULK,
        provider="openrouter",
        model="test/model",
        input_cost_per_million=0.10,
        output_cost_per_million=0.20,
    )
    cost = c.estimated_cost(1000, 500)
    assert cost is not None
    # (1000 * 0.10 + 500 * 0.20) / 1_000_000 = 0.0002
    assert abs(cost - 0.0002) < 1e-9


def test_model_candidate_unknown_cost():
    c = ModelCandidate(tier=TIER_CHEAP_BULK, provider="openrouter", model="test/model")
    assert c.estimated_cost(1000, 500) is None
