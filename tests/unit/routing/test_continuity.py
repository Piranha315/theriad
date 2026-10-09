"""
Tests for theriad/routing/continuity.py and the continuity integration in
theriad/models/policy.py.

Spec requirements tested:
- short prompts do not route to long-context models (context filtering)
- large prompts → smallest sufficient context bucket
- huge prompts → compression_or_chunking_recommended
- same conversation does not FORCE same model (bonus only, not hard pin)
- same task family can apply continuity bonus
- different task family → zero continuity bonus
- Aggressive Savings switches to cheaper model despite previous model
- Fast-ish policy (aggressive_savings) switches away when cost difference large
- route diagnostics explain continuity decision
- continuity bonus does not override context fit
- continuity bonus does not override tier floor (enforced by resolver)
"""
from __future__ import annotations

import pytest

from theriad.routing.continuity import (
    ContinuitySignal,
    ContinuityScore,
    _BONUS_WEIGHT_BY_POLICY,
    score_continuity,
)
from theriad.models.policy import PolicyResolution, PolicyResolver
from theriad.models.registry import ModelRegistry
from theriad.models.tiers import (
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_CHEAP_BULK,
    ModelCandidate,
)
from theriad.routing.context import ContextRequirementEstimator


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_registry(*candidates: ModelCandidate) -> ModelRegistry:
    """Build a minimal registry with the given candidates."""
    tier_map: dict[str, list[ModelCandidate]] = {}
    for c in candidates:
        tier_map.setdefault(c.tier, []).append(c)
    return ModelRegistry(tier_map)


def _candidate(
    tier: str,
    model: str,
    *,
    role: str = "primary",
    input_cost: float = 0.10,
    output_cost: float = 0.10,
    context_window: int | None = 32_000,
    quality: float = 0.70,
) -> ModelCandidate:
    return ModelCandidate(
        tier=tier,
        provider="openrouter",
        model=model,
        role=role,
        input_cost_per_million=input_cost,
        output_cost_per_million=output_cost,
        context_window=context_window,
        quality_prior=quality,
    )


# ── score_continuity ──────────────────────────────────────────────────────────


class TestScoreContinuity:
    def test_no_previous_model_returns_zero(self):
        signal = ContinuitySignal(previous_model=None, previous_tier=None)
        cs = score_continuity(signal, "some-model", "balanced")
        assert cs.bonus == 0.0
        assert not cs.applied
        assert cs.reason == "no_prior_model"

    def test_different_model_returns_zero(self):
        signal = ContinuitySignal(
            previous_model="model-a",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
        )
        cs = score_continuity(signal, "model-b", "balanced")
        assert cs.bonus == 0.0
        assert not cs.applied
        assert cs.reason == "different_model"

    def test_same_model_different_task_family_returns_zero(self):
        signal = ContinuitySignal(
            previous_model="model-a",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=False,
        )
        cs = score_continuity(signal, "model-a", "balanced")
        assert cs.bonus == 0.0
        assert not cs.applied
        assert "different_task_family" in cs.reason

    def test_same_model_same_family_same_type_full_bonus(self):
        signal = ContinuitySignal(
            previous_model="model-a",
            previous_tier=TIER_CHEAP_CODE,
            previous_task_type="code",
            current_task_type="code",
            same_task_family=True,
        )
        cs = score_continuity(signal, "model-a", "balanced")
        assert cs.applied
        assert cs.bonus == _BONUS_WEIGHT_BY_POLICY["balanced"]
        assert "same_task_type" in cs.reason

    def test_same_model_same_family_different_type_half_bonus(self):
        signal = ContinuitySignal(
            previous_model="model-a",
            previous_tier=TIER_CHEAP_CODE,
            previous_task_type="code",
            current_task_type="debug",
            same_task_family=True,
        )
        cs = score_continuity(signal, "model-a", "balanced")
        assert cs.applied
        expected = _BONUS_WEIGHT_BY_POLICY["balanced"] * 0.5
        assert abs(cs.bonus - expected) < 1e-9
        assert "same_task_family_different_type" in cs.reason

    def test_aggressive_savings_bonus_is_small(self):
        """Aggressive savings bonus must not flip a cost-driven decision."""
        signal = ContinuitySignal(
            previous_model="expensive-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
            previous_task_type="code",
            current_task_type="code",
        )
        cs = score_continuity(signal, "expensive-model", "aggressive_savings")
        assert cs.bonus <= 0.05  # very small by design

    def test_unknown_policy_uses_default_weight(self):
        signal = ContinuitySignal(
            previous_model="m",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
        )
        cs = score_continuity(signal, "m", "unknown_policy")
        assert cs.weight > 0.0  # default weight is positive


# ── PolicyResolver with continuity signal ─────────────────────────────────────


class TestPolicyResolverContinuity:
    def _two_candidate_registry(self) -> tuple[ModelRegistry, ModelCandidate, ModelCandidate]:
        prev = _candidate(TIER_CHEAP_CODE, "prev-model", input_cost=0.10, quality=0.70)
        alt = _candidate(
            TIER_CHEAP_CODE, "alt-model", role="fallback", input_cost=0.08, quality=0.65
        )
        registry = _make_registry(prev, alt)
        return registry, prev, alt

    def test_no_continuity_signal_uses_policy_baseline(self):
        registry, prev, alt = self._two_candidate_registry()
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced")
        assert resolution.model == "prev-model"  # primary wins
        assert not resolution.continuity_bonus_applied
        assert resolution.reason_for_stay_or_switch == "fresh_routing"

    def test_continuity_bonus_applied_same_model(self):
        """When previous model matches a candidate and task family matches,
        the bonus is applied and the flag is True."""
        registry, prev, alt = self._two_candidate_registry()
        signal = ContinuitySignal(
            previous_model="prev-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
            previous_task_type="code",
            current_task_type="code",
        )
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", continuity_signal=signal)
        assert resolution.continuity_bonus_applied
        assert resolution.continuity_bonus_weight > 0.0
        assert "continuity_bonus" in resolution.reason_for_stay_or_switch

    def test_same_conversation_does_not_force_same_model(self):
        """Continuity is a soft bonus: if a cheaper model scores better overall
        (aggressive_savings), the bonus must not be large enough to override it."""
        # cheap-model has much lower cost → scores higher under aggressive_savings
        cheap = _candidate(TIER_CHEAP_CODE, "cheap-model", input_cost=0.01, role="fallback")
        expensive = _candidate(TIER_CHEAP_CODE, "expensive-model", input_cost=5.00)
        registry = _make_registry(expensive, cheap)

        # Previous model was expensive; same task family
        signal = ContinuitySignal(
            previous_model="expensive-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
            previous_task_type="code",
            current_task_type="code",
        )
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(
            TIER_CHEAP_CODE, "aggressive_savings", continuity_signal=signal
        )
        # cheap model must win — cost difference dominates the tiny continuity bonus
        assert resolution.model == "cheap-model", (
            f"Expected cheap-model but got {resolution.model}; "
            "continuity bonus must not override a large cost difference"
        )

    def test_different_task_family_no_continuity_bonus(self):
        registry, prev, alt = self._two_candidate_registry()
        signal = ContinuitySignal(
            previous_model="prev-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=False,
        )
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", continuity_signal=signal)
        assert not resolution.continuity_bonus_applied

    def test_continuity_diagnostics_populated(self):
        registry, prev, alt = self._two_candidate_registry()
        signal = ContinuitySignal(
            previous_model="prev-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
        )
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", continuity_signal=signal)
        # Diagnostics should be non-empty
        assert resolution.reason_for_stay_or_switch != ""

    def test_switched_model_reason_explains_switch(self):
        """When bonus applied but previous model still loses, reason names the switch."""
        # cheap wins despite continuity bonus on expensive
        cheap = _candidate(TIER_CHEAP_CODE, "cheap-model", input_cost=0.001, role="fallback")
        expensive = _candidate(TIER_CHEAP_CODE, "expensive-model", input_cost=100.0)
        registry = _make_registry(expensive, cheap)
        signal = ContinuitySignal(
            previous_model="expensive-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
            previous_task_type="code",
            current_task_type="code",
        )
        resolver = PolicyResolver(registry)
        resolution = resolver.resolve(
            TIER_CHEAP_CODE, "aggressive_savings", continuity_signal=signal
        )
        assert resolution.model == "cheap-model"
        assert "switched_model" in resolution.reason_for_stay_or_switch


# ── PolicyResolver with context estimate ──────────────────────────────────────


class TestPolicyResolverContext:
    def test_small_context_does_not_require_large_window_model(self):
        """A short prompt must not be forced onto a large-context model."""
        small_ctx = _candidate(TIER_CHEAP_CODE, "small-ctx", context_window=8_000)
        large_ctx = _candidate(TIER_CHEAP_CODE, "large-ctx", context_window=128_000, role="fallback")
        registry = _make_registry(small_ctx, large_ctx)
        resolver = PolicyResolver(registry)
        estimate = ContextRequirementEstimator().estimate("hi")
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", context_estimate=estimate)
        # Both pass the filter; primary wins under balanced
        assert resolution.model == "small-ctx"

    def test_large_prompt_routes_to_sufficient_context_model(self):
        """Candidates whose context_window is too small for the prompt are excluded."""
        small_ctx = _candidate(
            TIER_CHEAP_CODE, "small-ctx", context_window=4_000, role="primary"
        )
        large_ctx = _candidate(
            TIER_CHEAP_CODE, "large-ctx", context_window=128_000, role="fallback"
        )
        registry = _make_registry(small_ctx, large_ctx)
        resolver = PolicyResolver(registry)

        # ~25,000 tokens → MEDIUM bucket; small_ctx (4K) excluded
        estimate = ContextRequirementEstimator().estimate("x" * 100_000)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", context_estimate=estimate)
        assert resolution.model == "large-ctx"

    def test_huge_prompt_sets_chunking_recommendation(self):
        """Huge prompts set compression_or_chunking_recommended in resolution."""
        candidate = _candidate(TIER_CHEAP_CODE, "big-model", context_window=None)
        registry = _make_registry(candidate)
        resolver = PolicyResolver(registry)
        estimate = ContextRequirementEstimator().estimate("x" * 600_000)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", context_estimate=estimate)
        assert resolution.compression_or_chunking_recommended

    def test_context_diagnostics_populated(self):
        candidate = _candidate(TIER_CHEAP_CODE, "m", context_window=32_000)
        registry = _make_registry(candidate)
        resolver = PolicyResolver(registry)
        estimate = ContextRequirementEstimator().estimate("hello world")
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", context_estimate=estimate)
        assert resolution.selected_context_bucket is not None
        assert resolution.context_fit_reason is not None
        assert resolution.selected_model_context_limit == 32_000

    def test_context_window_none_passes_filter(self):
        """Unknown context_window treated as sufficient — not excluded."""
        unknown_ctx = _candidate(TIER_CHEAP_CODE, "unknown", context_window=None)
        registry = _make_registry(unknown_ctx)
        resolver = PolicyResolver(registry)
        estimate = ContextRequirementEstimator().estimate("x" * 500_000)
        resolution = resolver.resolve(TIER_CHEAP_CODE, "balanced", context_estimate=estimate)
        assert resolution.model == "unknown"

    def test_continuity_does_not_override_context_fit(self):
        """Continuity bonus on a model with insufficient context must not select it."""
        small_ctx = _candidate(
            TIER_CHEAP_CODE, "prev-model", context_window=4_000, role="primary"
        )
        large_ctx = _candidate(
            TIER_CHEAP_CODE, "new-model", context_window=128_000, role="fallback"
        )
        registry = _make_registry(small_ctx, large_ctx)
        resolver = PolicyResolver(registry)

        # Large prompt — prev-model (4K window) must be excluded by context filter
        estimate = ContextRequirementEstimator().estimate("x" * 100_000)
        signal = ContinuitySignal(
            previous_model="prev-model",
            previous_tier=TIER_CHEAP_CODE,
            same_task_family=True,
            previous_task_type="code",
            current_task_type="code",
        )
        resolution = resolver.resolve(
            TIER_CHEAP_CODE, "balanced",
            context_estimate=estimate,
            continuity_signal=signal,
        )
        # new-model must win — context filter excludes prev-model first
        assert resolution.model == "new-model"


# ── StickyRouteCache.get_continuity_signal ────────────────────────────────────


class TestStickyRouteCacheSignal:
    def test_no_cached_entry_returns_none(self):
        from theriad.routing.sticky import StickyRouteCache
        cache = StickyRouteCache()
        result = cache.get_continuity_signal("conv-1", TIER_CHEAP_CODE)
        assert result is None

    def test_cached_entry_returns_signal(self):
        from theriad.routing.sticky import StickyRouteCache
        cache = StickyRouteCache()
        cache.set(
            "conv-1",
            selected_model="qwen/qwen3",
            selected_tier=TIER_CHEAP_CODE,
            provider="openrouter",
            reason="first_call",
        )
        signal = cache.get_continuity_signal("conv-1", TIER_CHEAP_CODE)
        assert signal is not None
        assert signal.previous_model == "qwen/qwen3"
        assert signal.previous_tier == TIER_CHEAP_CODE
        assert signal.same_task_family is True

    def test_major_tier_change_clears_cache_and_returns_none(self):
        from theriad.routing.sticky import StickyRouteCache
        cache = StickyRouteCache()
        cache.set(
            "conv-1",
            selected_model="cheap-model",
            selected_tier=TIER_CHEAP_BULK,
            provider="openrouter",
            reason="first_call",
        )
        # TIER_CHEAP_BULK → TIER_CODE_STRONG is a major change (rank delta = 2)
        signal = cache.get_continuity_signal("conv-1", TIER_CODE_STRONG)
        assert signal is None
        # Cache should have been cleared
        assert cache.get("conv-1") is None

    def test_minor_tier_change_preserves_cache(self):
        from theriad.routing.sticky import StickyRouteCache
        cache = StickyRouteCache()
        cache.set(
            "conv-1",
            selected_model="code-model",
            selected_tier=TIER_CHEAP_CODE,
            provider="openrouter",
            reason="first_call",
        )
        # TIER_CHEAP_CODE → TIER_CODE_STRONG is delta=1 (minor)
        signal = cache.get_continuity_signal("conv-1", TIER_CODE_STRONG)
        assert signal is not None
        assert signal.same_task_family is True
        # Cache entry should still be live
        assert cache.get("conv-1") is not None
