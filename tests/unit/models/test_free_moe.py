"""Tests for the opt-in free_moe_experimental routing policy."""
from __future__ import annotations

from theriad.models.free_moe import FreeMoeResolver, route_free_moe
from theriad.models.policy import POLICY_TRADEOFF, VALID_POLICIES, PolicyResolution, PolicyResolver
from theriad.models.registry import DEFAULT_REGISTRY
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_FREE_CODE,
    TIER_FREE_GENERAL,
    TIER_FREE_MOE_TIERS,
    TIER_FREE_SUMMARIZE,
    TIER_FREE_VERIFY,
    VALID_TIERS,
)


def test_code_signals_route_to_code_pool():
    decision = route_free_moe("Please fix this: def broken(value):")
    assert decision.tier == TIER_FREE_CODE
    assert decision.pool == "code_free"
    assert decision.matched_signal == "def"


def test_verify_signals_route_to_verify_pool():
    decision = route_free_moe("Fact-check this claim and cite a source.")
    assert decision.tier == TIER_FREE_VERIFY
    assert decision.pool == "verify_free"


def test_summarize_signals_route_to_summarize_pool():
    decision = route_free_moe("Summarize the following meeting notes.")
    assert decision.tier == TIER_FREE_SUMMARIZE
    assert decision.pool == "summarize_free"


def test_long_message_routes_to_summarize_pool():
    decision = route_free_moe("A long document sentence. " * 40)
    assert decision.tier == TIER_FREE_SUMMARIZE
    assert decision.matched_signal == "long_message"


def test_long_code_message_still_routes_to_code_pool():
    decision = route_free_moe("def parse(value):\n    return value\n" + ("context " * 120))
    assert decision.tier == TIER_FREE_CODE


def test_default_routes_to_general_pool():
    decision = route_free_moe("What should I cook for dinner?")
    assert decision.tier == TIER_FREE_GENERAL
    assert decision.pool == "general_free"
    assert decision.confidence == 0.7


def test_free_moe_resolver_returns_policy_resolution():
    resolution = FreeMoeResolver().resolve("debug this Python error")
    assert isinstance(resolution, PolicyResolution)
    assert resolution.policy == "free_moe_experimental"
    assert resolution.tier == TIER_FREE_CODE


def test_free_moe_resolver_selects_free_model():
    resolution = FreeMoeResolver().resolve("debug this Python error")
    assert resolution.selected.input_cost_per_million == 0.0
    assert resolution.selected.output_cost_per_million == 0.0
    assert resolution.model.endswith(":free")


def test_policy_resolver_threads_message_to_free_moe():
    resolution = PolicyResolver().resolve(
        TIER_CHEAP_BULK,
        "free_moe_experimental",
        message="Please summarize these notes",
    )
    assert resolution.tier == TIER_FREE_SUMMARIZE
    assert resolution.policy == "free_moe_experimental"
    assert resolution.cost_quality_tradeoff == 10


def test_free_moe_not_active_by_default():
    resolution = PolicyResolver().resolve(TIER_CHEAP_BULK, "balanced")
    assert resolution.tier == TIER_CHEAP_BULK
    assert resolution.policy == "balanced"


def test_free_moe_experimental_in_valid_policies():
    assert "free_moe_experimental" in VALID_POLICIES
    assert POLICY_TRADEOFF["free_moe_experimental"] == 10


def test_free_moe_tiers_in_valid_tiers():
    assert TIER_FREE_MOE_TIERS <= VALID_TIERS


def test_every_free_moe_candidate_is_zero_cost_and_uses_free_slug():
    for tier in TIER_FREE_MOE_TIERS:
        for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier):
            assert candidate.model.endswith(":free")
            assert candidate.input_cost_per_million == 0.0
            assert candidate.output_cost_per_million == 0.0
