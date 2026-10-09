"""Cross-tier escalation and quality-constrained selection.

The behaviour under test is the one described in theriad/models/unified.py:
widen the candidate pool up the tier ladder, then take the *cheapest* model
whose measured P(success) on this prompt clears SUCCESS_THRESHOLD.
"""
from __future__ import annotations

import pytest

from theriad.models.registry import DEFAULT_REGISTRY
from theriad.models.tiers import ModelCandidate
from theriad.models.unified import (
    escalation_candidates,
    select_by_success,
)


def _candidate(model: str, tier: str, cost: float) -> ModelCandidate:
    return ModelCandidate(
        tier=tier,
        provider="test",
        model=model,
        input_cost_per_million=cost,
        output_cost_per_million=cost,
        quality_prior=0.8,
    )


# ── Ladder construction ───────────────────────────────────────────────────────


def test_general_lane_climbs_to_premium_by_default():
    tiers = {c.tier for c in escalation_candidates("cheap_bulk", DEFAULT_REGISTRY)}
    assert "cheap_bulk" in tiers
    assert "premium" in tiers


def test_frontier_is_not_reachable_by_default():
    """tiers.py records frontier as entered deliberately, not routinely."""
    tiers = {c.tier for c in escalation_candidates("cheap_bulk", DEFAULT_REGISTRY)}
    assert "frontier" not in tiers

    explicit = escalation_candidates("cheap_bulk", DEFAULT_REGISTRY, ceiling="frontier")
    assert "frontier" in {c.tier for c in explicit}


def test_lanes_do_not_cross():
    """A bulk request must not escalate into a code tier: that changes task
    domain rather than climbing quality."""
    general = {c.tier for c in escalation_candidates("cheap_bulk", DEFAULT_REGISTRY)}
    assert "cheap_code" not in general
    assert "code_strong" not in general

    code = {c.tier for c in escalation_candidates("cheap_code", DEFAULT_REGISTRY)}
    assert "cheap_bulk" not in code
    assert "code_strong" in code


def test_unknown_tier_yields_only_itself():
    out = escalation_candidates("free_moe_general", DEFAULT_REGISTRY)
    assert {c.tier for c in out} == {"free_moe_general"}


def test_candidates_are_deduplicated_across_rungs():
    """Several models appear in more than one tier; the pool must not double
    count them, or the cheapest-first scan can return a duplicate."""
    out = escalation_candidates("cheap_bulk", DEFAULT_REGISTRY)
    models = [c.model for c in out]
    assert len(models) == len(set(models))


# ── Selection ─────────────────────────────────────────────────────────────────

CHEAP = _candidate("cheap", "cheap_bulk", 0.0)
MID = _candidate("mid", "premium", 1.0)
STRONG = _candidate("strong", "premium", 10.0)
POOL = [CHEAP, MID, STRONG]
ABILITIES = {"cheap": 1.0, "mid": 4.0, "strong": 8.0}


def test_easy_request_stays_on_the_cheapest_model():
    # difficulty -5 => P(cheap) = sigmoid(6) = 0.9975, clears 0.9.
    assert select_by_success(POOL, -5.0, ABILITIES, threshold=0.9) is CHEAP


def test_hard_request_escalates_to_the_cheapest_model_that_clears():
    # difficulty 1.5 => cheap sigmoid(-0.5)=0.38 fails, mid sigmoid(2.5)=0.924
    # clears. MID, not STRONG: the bar is a floor, not a maximisation.
    assert select_by_success(POOL, 1.5, ABILITIES, threshold=0.9) is MID


def test_impossible_request_falls_back_to_the_most_capable():
    """Nothing clears the bar, so the best available shot is the honest answer
    rather than the cheapest guess."""
    assert select_by_success(POOL, 50.0, ABILITIES, threshold=0.9) is STRONG


def test_returns_none_when_no_ability_is_known():
    """Without a measured ability there is no success probability, and
    inventing one would defeat the purpose. The caller must fall back to
    expected value instead."""
    assert select_by_success(POOL, 0.0, {}, threshold=0.9) is None


def test_partial_abilities_only_consider_measured_candidates():
    chosen = select_by_success(POOL, 1.5, {"strong": 8.0}, threshold=0.9)
    assert chosen is STRONG


def test_threshold_reads_the_module_constant_at_call_time():
    """Regression: SUCCESS_THRESHOLD was a default argument, so it bound at
    definition and a sweep over it silently produced identical runs."""
    import theriad.models.unified as unified

    original = unified.SUCCESS_THRESHOLD
    try:
        unified.SUCCESS_THRESHOLD = 0.99
        strict = select_by_success(POOL, 1.5, ABILITIES)
        unified.SUCCESS_THRESHOLD = 0.10
        loose = select_by_success(POOL, 1.5, ABILITIES)
    finally:
        unified.SUCCESS_THRESHOLD = original

    assert strict is STRONG
    assert loose is CHEAP


# ── Resolver integration ──────────────────────────────────────────────────────


def test_escalation_can_be_switched_off():
    """The opt-out must reproduce within-tier selection exactly."""
    from theriad.models.policy import PolicyResolver

    resolver = PolicyResolver()
    prompt = "Derive the second-order perturbation correction and justify each step."
    off = resolver.resolve("cheap_bulk", "balanced", message=prompt, escalation=False)
    assert off.tier == "cheap_bulk"
    assert off.selected.tier == "cheap_bulk"


def test_escalation_is_inert_without_a_message():
    """No prompt means no difficulty estimate, so there is nothing to condition
    on and the resolver must not guess."""
    from theriad.models.policy import PolicyResolver

    resolved = PolicyResolver().resolve("cheap_bulk", "balanced")
    assert resolved.selected.tier == "cheap_bulk"


def test_resolution_reports_the_tier_actually_used():
    """Per-tier output caps and timeouts key off PolicyResolution.tier. Keeping
    the routed tier after escalating would cap a premium model at cheap_bulk's
    1200 tokens and truncate it."""
    from theriad.models.policy import PolicyResolver

    resolver = PolicyResolver()
    prompt = "x" * 4000
    resolved = resolver.resolve("cheap_bulk", "balanced", message=prompt)
    assert resolved.tier == resolved.selected.tier


def test_tier_local_winner_becomes_the_first_fallback():
    """If the escalated model fails, the next attempt should be whatever the
    routed tier would have chosen."""
    from theriad.models.policy import PolicyResolver
    from theriad.routing.difficulty import load_difficulty_model

    if load_difficulty_model().is_empty:
        pytest.skip("no difficulty model installed")

    resolver = PolicyResolver()
    prompt = "Explain the mechanism and justify each step in detail.\n" * 40
    resolved = resolver.resolve("cheap_bulk", "balanced", message=prompt)
    if resolved.selected.tier == "cheap_bulk":
        pytest.skip("this prompt did not escalate; nothing to assert")
    assert resolved.fallback_chain
    assert resolved.fallback_chain[0].tier == "cheap_bulk"
