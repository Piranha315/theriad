from theriad.models.policy import PolicyResolver
from theriad.models.preferences import (
    RoutingPreferences,
    estimated_price_multiplier,
    legacy_policy_for_preferences,
)
from theriad.models.registry import ModelRegistry
from theriad.models.tiers import TIER_CHEAP_BULK, ModelCandidate


def _candidate(model: str, cost: float, quality: float, role: str) -> ModelCandidate:
    return ModelCandidate(
        tier=TIER_CHEAP_BULK,
        provider="openrouter",
        model=model,
        role=role,
        input_cost_per_million=cost,
        output_cost_per_million=cost,
        quality_prior=quality,
    )


def test_standard_preferences_are_one_x():
    assert estimated_price_multiplier(50, 50) == 1.0


def test_price_estimate_moves_with_each_axis():
    standard = estimated_price_multiplier(50, 50)
    assert estimated_price_multiplier(10, 50) < standard
    assert estimated_price_multiplier(90, 50) > standard
    assert estimated_price_multiplier(50, 10) < standard
    assert estimated_price_multiplier(50, 90) > standard


def test_preferences_preserve_legacy_clients():
    assert (
        RoutingPreferences.resolve(speed=None, quality=None, legacy_policy="quality_first").quality
        == 90
    )
    assert legacy_policy_for_preferences(20, 20) == "aggressive_savings"


def test_quality_dial_no_longer_changes_selected_model():
    """The dial was removed (D-054), not retuned.

    It previously moved selection between `cheap` and `strong`, which meant the
    same prompt got a different model depending on a slider most callers never
    touched. The preference values are still accepted so existing requests do
    not break; they simply have no effect on which model is chosen.
    """
    cheap = _candidate("cheap", 0.1, 0.55, "primary")
    strong = _candidate("strong", 1.0, 0.95, "fallback")
    resolver = PolicyResolver(ModelRegistry({TIER_CHEAP_BULK: [cheap, strong]}))

    low = resolver.resolve(TIER_CHEAP_BULK, speed_preference=50, quality_preference=10)
    high = resolver.resolve(TIER_CHEAP_BULK, speed_preference=50, quality_preference=95)
    assert low.model == high.model

    # And the model chosen is the one the unified rule prefers.
    from theriad.models.unified import expected_value

    best = max((cheap, strong), key=lambda c: expected_value(c, None, TIER_CHEAP_BULK))
    assert low.model == best.model
