"""Tests for PolicyResolver."""
from __future__ import annotations

import pytest

from theriad.models.policy import VALID_POLICIES, PolicyResolver
from theriad.models.registry import ModelRegistry
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    ModelCandidate,
)


def _make_registry(*candidates: ModelCandidate) -> ModelRegistry:
    """Build a registry from a flat list, grouped by tier."""
    by_tier: dict[str, list[ModelCandidate]] = {}
    for c in candidates:
        by_tier.setdefault(c.tier, []).append(c)
    return ModelRegistry(by_tier)


FREE = ModelCandidate(
    tier=TIER_CHEAP_BULK, provider="openrouter", model="free/model",
    role="primary", input_cost_per_million=0.0, output_cost_per_million=0.0,
    quality_prior=0.55,
)
CHEAP = ModelCandidate(
    tier=TIER_CHEAP_BULK, provider="openrouter", model="cheap/model",
    role="fallback", input_cost_per_million=0.10, output_cost_per_million=0.20,
    quality_prior=0.70,
)
EXPENSIVE = ModelCandidate(
    tier=TIER_CHEAP_BULK, provider="openrouter", model="expensive/model",
    role="fallback", input_cost_per_million=5.00, output_cost_per_million=15.00,
    quality_prior=0.90,
)


# ── PolicyResolver basics ─────────────────────────────────────────────────────

def test_resolver_uses_default_registry():
    resolver = PolicyResolver()
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    assert resolution.model
    assert resolution.provider
    assert resolution.tier == TIER_CHEAP_BULK


def test_valid_policies():
    assert {
        "aggressive_savings",
        "balanced",
        "quality_first",
        "free_moe_experimental",
    } == VALID_POLICIES


def _DEAD_test_unknown_policy_falls_back_to_balanced():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    # balanced picks role='primary' = FREE
    resolution = resolver.resolve(TIER_CHEAP_BULK, "unknown_policy")
    assert resolution.policy == "balanced"
    assert resolution.model == "free/model"


def test_unknown_tier_falls_back_to_cheap_bulk():
    resolver = PolicyResolver()
    resolution = resolver.resolve("nonexistent_tier", "balanced")
    assert resolution.tier == TIER_CHEAP_BULK


# ── balanced policy ───────────────────────────────────────────────────────────

def _DEAD_test_balanced_picks_primary():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    assert resolution.model == "free/model"  # role='primary'


def _DEAD_test_balanced_fallback_chain_excludes_selected():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    fallback_models = [m for _, m in resolution.fallback_tuples()]
    assert "free/model" not in fallback_models


# ── aggressive_savings policy ─────────────────────────────────────────────────

def _DEAD_test_aggressive_savings_picks_free_at_high_confidence():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "aggressive_savings", confidence=0.9)
    assert resolution.model == "free/model"


def test_aggressive_savings_skips_free_at_low_confidence():
    # Build registry where the cheapest paid is CHEAP
    only_paid = [
        ModelCandidate(tier=TIER_CHEAP_BULK, provider="openrouter", model="cheap/paid",
                       role="primary", input_cost_per_million=0.10, output_cost_per_million=0.20,
                       quality_prior=0.70),
        ModelCandidate(tier=TIER_CHEAP_BULK, provider="openrouter", model="expensive/paid",
                       role="fallback", input_cost_per_million=5.00, output_cost_per_million=15.00,
                       quality_prior=0.90),
    ]
    registry = _make_registry(*only_paid)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "aggressive_savings", confidence=0.5)
    assert resolution.model == "cheap/paid"  # cheapest paid


# ── quality_first policy ──────────────────────────────────────────────────────

def _DEAD_test_quality_first_picks_highest_quality():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "quality_first")
    assert resolution.model == "expensive/model"  # quality_prior=0.90


def _DEAD_test_quality_first_unknown_prior_treated_as_zero():
    no_prior = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="no-prior/model",
        role="primary", quality_prior=None,
    )
    with_prior = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="has-prior/model",
        role="fallback", quality_prior=0.5,
    )
    registry = _make_registry(no_prior, with_prior)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "quality_first")
    assert resolution.model == "has-prior/model"


# ── PolicyResolution helpers ──────────────────────────────────────────────────

def test_resolution_fallback_tuples():
    registry = _make_registry(FREE, CHEAP, EXPENSIVE)
    resolver = PolicyResolver(registry)
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    tuples = resolution.fallback_tuples()
    assert all(len(t) == 2 for t in tuples)
    assert all(isinstance(t[0], str) and isinstance(t[1], str) for t in tuples)


def test_resolution_provider_and_model():
    resolver = PolicyResolver()
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    assert resolution.provider == resolution.selected.provider
    assert resolution.model == resolution.selected.model


# ── Unified router (D-054) ────────────────────────────────────────────────────
#
# The tests above prefixed with _DEAD_ asserted per-policy selection:
# balanced picks the primary, quality_first picks the highest prior,
# aggressive_savings picks the cheapest. That behaviour was removed on purpose —
# three policies were three answers to a question the user should never have
# been asked, and they disagreed with each other on the same prompt. They are
# kept, renamed, as a record of what the router used to promise.
#
# What replaces them is a single invariant: the policy argument does not change
# the outcome.

from theriad.models.unified import RoutingConstraints, expected_value  # noqa: E402


def test_policy_argument_no_longer_changes_selection():
    resolver = PolicyResolver(_make_registry(FREE, CHEAP, EXPENSIVE))
    picks = {
        p: resolver.resolve(TIER_CHEAP_BULK, p).model
        for p in ("balanced", "aggressive_savings", "quality_first")
    }
    assert len(set(picks.values())) == 1, f"policy still matters: {picks}"


def test_unknown_policy_is_harmless():
    """It could not select differently even if it wanted to."""
    resolver = PolicyResolver(_make_registry(FREE, CHEAP, EXPENSIVE))
    assert (
        resolver.resolve(TIER_CHEAP_BULK, "not_a_policy").model
        == resolver.resolve(TIER_CHEAP_BULK, "balanced").model
    )


def test_selection_maximises_expected_value():
    resolver = PolicyResolver(_make_registry(FREE, CHEAP, EXPENSIVE))
    selected = resolver.resolve(TIER_CHEAP_BULK, "balanced").selected
    best = max(
        (FREE, CHEAP, EXPENSIVE),
        key=lambda c: expected_value(c, None, TIER_CHEAP_BULK),
    )
    assert selected.model == best.model


def test_fallback_chain_excludes_the_selected_model():
    resolver = PolicyResolver(_make_registry(FREE, CHEAP, EXPENSIVE))
    resolution = resolver.resolve(TIER_CHEAP_BULK, "balanced")
    assert resolution.model not in [m for _, m in resolution.fallback_tuples()]


def test_fallback_chain_is_ordered_by_expected_value():
    """A failing primary should degrade to the next best trade, not the next
    line in the registry."""
    resolver = PolicyResolver(_make_registry(FREE, CHEAP, EXPENSIVE))
    chain = resolver.resolve(TIER_CHEAP_BULK, "balanced").fallback_chain
    evs = [expected_value(c, None, TIER_CHEAP_BULK) for c in chain]
    assert evs == sorted(evs, reverse=True)


# ── The one remaining option: vendor origin ───────────────────────────────────

def test_origin_constraint_excludes_a_vendor():
    cn = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="cn/cheap",
        origin="CN", role="primary",
        input_cost_per_million=0.01, output_cost_per_million=0.01,
        quality_prior=0.90,
    )
    us = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="us/dearer",
        origin="US", role="fallback",
        input_cost_per_million=1.0, output_cost_per_million=1.0,
        quality_prior=0.80,
    )
    resolver = PolicyResolver(_make_registry(cn, us))
    assert resolver.resolve(TIER_CHEAP_BULK, "balanced").model == "cn/cheap"
    assert resolver.resolve(
        TIER_CHEAP_BULK, "balanced", constraints=RoutingConstraints.exclude_china()
    ).model == "us/dearer"


def test_unattributed_origin_fails_an_active_constraint():
    """A model we cannot attribute must not silently satisfy a restriction
    someone set deliberately."""
    unknown = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="mystery/model",
        role="primary", input_cost_per_million=0.01, output_cost_per_million=0.01,
        quality_prior=0.90,
    )
    resolver = PolicyResolver(_make_registry(unknown))
    assert resolver.resolve(TIER_CHEAP_BULK, "balanced").model == "mystery/model"
    with pytest.raises(RuntimeError, match="origin constraint"):
        resolver.resolve(
            TIER_CHEAP_BULK, "balanced", constraints=RoutingConstraints.exclude_china()
        )


def test_constraint_is_honoured_over_availability():
    """Refusing is correct: degrading to an excluded vendor would break the
    guarantee invisibly."""
    cn_only = ModelCandidate(
        tier=TIER_CHEAP_BULK, provider="openrouter", model="cn/only",
        origin="CN", role="primary",
        input_cost_per_million=0.01, output_cost_per_million=0.01,
        quality_prior=0.9,
    )
    resolver = PolicyResolver(_make_registry(cn_only))
    with pytest.raises(RuntimeError, match="origin constraint"):
        resolver.resolve(
            TIER_CHEAP_BULK, "balanced", constraints=RoutingConstraints.exclude_china()
        )
