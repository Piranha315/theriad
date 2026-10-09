"""Calibration: grid → router parameters.

The interesting cases here are the refusals. A calibration that silently
produces numbers from an unfittable grid is worse than no calibration, because
at the router the two are indistinguishable.
"""
from __future__ import annotations

import pytest

from theriad.eval.calibrate import (
    MIN_AXIS_TRIALS,
    AxisMeasurement,
    best_oracle_value,
    QualityScale,
    ScaleFitError,
    fit_quality_scale,
    measure_axes,
    quality_profiles,
    router_value_from_oracle_value,
)
from theriad.eval.corpus import EvalItem
from theriad.eval.grid import OutcomeGrid
from theriad.models.tiers import ModelCandidate


class FakeRegistry:
    """Duck-typed stand-in so tests do not depend on the shipped registry."""

    def __init__(self, candidates: list[ModelCandidate]):
        self._candidates = candidates

    def all_tiers(self) -> list[str]:
        return sorted({c.tier for c in self._candidates})

    def candidates_for_tier(self, tier: str) -> list[ModelCandidate]:
        return [c for c in self._candidates if c.tier == tier]


def _candidate(model: str, prior: float, source: str) -> ModelCandidate:
    return ModelCandidate(
        tier="premium", provider="openrouter", model=model,
        quality_prior=prior, source=source,
        input_cost_per_million=1.0, output_cost_per_million=2.0,
    )


def _items(n: int, category: str = "reasoning") -> list[EvalItem]:
    return [
        EvalItem(
            id=f"{category}-{i}", category=category, prompt="q",
            answer="A", answer_type="multiple_choice", source="mmlu", vintage=2021,
        )
        for i in range(n)
    ]


def _fill(grid: OutcomeGrid, model: str, items: list[EvalItem], rate: float) -> None:
    """Record 5 trials per item at approximately ``rate``."""
    for idx, item in enumerate(items):
        for k in range(5):
            success = ((idx * 5 + k) % 100) < round(rate * 100)
            grid.record(model, item.id, success, cost_usd=0.001)


# ── Measurement ───────────────────────────────────────────────────────────────


def test_measure_axes_pools_by_axis_and_aggregate():
    items = _items(10, "reasoning") + _items(10, "code")
    grid = OutcomeGrid()
    _fill(grid, "m", items, 0.6)

    m = measure_axes(grid, items)["m"]

    assert m[None].trials == 100
    assert m["reasoning"].trials == 50
    assert m["code"].trials == 50
    assert m[None].successes == m["reasoning"].successes + m["code"].successes


def test_unmapped_category_contributes_to_aggregate_but_gets_no_axis():
    items = _items(10, "trivia-not-an-axis")
    grid = OutcomeGrid()
    _fill(grid, "m", items, 0.5)

    m = measure_axes(grid, items)["m"]

    assert m[None].trials == 50
    assert set(m) == {None}


def test_jeffreys_smoothing_keeps_a_sweep_off_1_0():
    m = AxisMeasurement(model="m", axis="code", trials=20, successes=20)
    assert m.rate < 1.0
    assert m.rate > 0.95


# ── Scale fitting ─────────────────────────────────────────────────────────────


def test_fit_recovers_a_known_affine_relationship():
    items = _items(20)
    grid = OutcomeGrid()
    # quality = 0.5 * rate + 0.5, evaluated at three separated rates.
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, items, rate)
    registry = FakeRegistry([
        _candidate("a", 0.6, "artificial_analysis_2026-08"),
        _candidate("b", 0.75, "artificial_analysis_2026-08"),
        _candidate("c", 0.9, "artificial_analysis_2026-08"),
    ])

    scale = fit_quality_scale(measure_axes(grid, items), registry)

    assert scale.slope == pytest.approx(0.5, abs=0.05)
    assert scale.intercept == pytest.approx(0.5, abs=0.05)
    assert scale.anchors == ["a", "b", "c"]


def test_hand_authored_priors_cannot_anchor_the_scale():
    items = _items(20)
    grid = OutcomeGrid()
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, items, rate)
    registry = FakeRegistry([
        _candidate("a", 0.6, "artificial_analysis_2026-08"),
        _candidate("b", 0.75, "openrouter_free_tier"),
        _candidate("c", 0.9, "openrouter_free_tier"),
    ])

    with pytest.raises(ScaleFitError, match="usable anchors"):
        fit_quality_scale(measure_axes(grid, items), registry)


def test_thin_coverage_does_not_anchor():
    items = _items(2)  # 10 trials each, below MIN_AXIS_TRIALS
    assert 2 * 5 < MIN_AXIS_TRIALS
    grid = OutcomeGrid()
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, items, rate)
    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])

    with pytest.raises(ScaleFitError, match="usable anchors"):
        fit_quality_scale(measure_axes(grid, items), registry)


def test_inverted_ordering_is_refused_rather_than_fitted():
    """Better published models scoring worse is a finding, not a scale."""
    items = _items(20)
    grid = OutcomeGrid()
    for model, rate in [("a", 0.8), ("b", 0.5), ("c", 0.2)]:
        _fill(grid, model, items, rate)
    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])

    with pytest.raises(ScaleFitError, match="not positive"):
        fit_quality_scale(measure_axes(grid, items), registry)


def test_no_spread_is_refused():
    items = _items(20)
    grid = OutcomeGrid()
    for model in ("a", "b", "c"):
        _fill(grid, model, items, 0.5)
    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])

    with pytest.raises(ScaleFitError, match="does not separate"):
        fit_quality_scale(measure_axes(grid, items), registry)


# ── Profiles ──────────────────────────────────────────────────────────────────


def test_profiles_land_on_the_registry_band_not_the_raw_rate():
    items = _items(40, "reasoning")
    grid = OutcomeGrid()
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, items, rate)
    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])
    measurements = measure_axes(grid, items)
    scale = fit_quality_scale(measurements, registry)

    profiles = quality_profiles(measurements, scale, registry)

    # Raw rates were 0.2/0.5/0.8; scores must sit on the priors' band instead.
    for model, prior in [("a", 0.6), ("b", 0.75), ("c", 0.9)]:
        assert profiles[model]["reasoning"] == pytest.approx(prior, abs=0.05)
    # Ordering is preserved.
    assert (
        profiles["a"]["reasoning"]
        < profiles["b"]["reasoning"]
        < profiles["c"]["reasoning"]
    )


def test_thin_axis_stays_near_the_prior():
    """25 trials should barely move a published index."""
    strong = _items(40, "reasoning")
    thin = _items(5, "code")
    items = strong + thin
    grid = OutcomeGrid()
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, strong, rate)
    _fill(grid, "b", thin, 1.0)  # 25 trials, perfect score

    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])
    measurements = measure_axes(grid, items)
    scale = fit_quality_scale(measurements, registry)
    profiles = quality_profiles(measurements, scale, registry)

    # A perfect 25-trial run maps well above the prior before shrinkage;
    # shrinkage must keep most of the weight on the published 0.75.
    assert 0.75 < profiles["b"]["code"] < 0.85


def test_axis_below_the_trial_floor_is_omitted_entirely():
    strong = _items(40, "reasoning")
    tiny = _items(2, "code")  # 10 trials
    items = strong + tiny
    grid = OutcomeGrid()
    for model, rate in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        _fill(grid, model, strong, rate)
    _fill(grid, "b", tiny, 1.0)

    registry = FakeRegistry([
        _candidate(m, p, "artificial_analysis_2026-08")
        for m, p in [("a", 0.6), ("b", 0.75), ("c", 0.9)]
    ])
    measurements = measure_axes(grid, items)
    scale = fit_quality_scale(measurements, registry)

    assert "code" not in quality_profiles(measurements, scale, registry)["b"]


# ── Units ─────────────────────────────────────────────────────────────────────


def _sweep(rows: list[tuple[float, float]]) -> list[dict]:
    return [
        {"value_correct": v, "summary": {"regret_per_1k_requests_usd": r}}
        for v, r in rows
    ]


def test_interior_minimum_is_identified():
    best, interior = best_oracle_value(
        _sweep([(0.01, 5.0), (0.05, 1.0), (0.10, 3.0), (0.50, 6.0)])
    )
    assert (best, interior) == (0.05, True)


def test_minimum_on_the_upper_edge_is_not_identified():
    """Regret still falling when the sweep ran out is not an optimum."""
    _, interior = best_oracle_value(
        _sweep([(0.01, 9.0), (0.05, 6.0), (0.10, 3.0), (0.50, 1.0)])
    )
    assert interior is False


def test_minimum_on_the_lower_edge_is_not_identified():
    _, interior = best_oracle_value(
        _sweep([(0.01, 1.0), (0.05, 3.0), (0.10, 6.0), (0.50, 9.0)])
    )
    assert interior is False


def test_ties_break_toward_the_cheaper_value():
    best, _ = best_oracle_value(
        _sweep([(0.01, 9.0), (0.05, 2.0), (0.10, 2.0), (0.50, 9.0)])
    )
    assert best == 0.05


def test_oracle_value_converts_into_per_1m_token_units():
    """V_router = V_oracle x 1e6 / (slope x tokens_per_call)."""
    scale = QualityScale(slope=0.5, intercept=0.5, anchors=["a", "b", "c"])

    v = router_value_from_oracle_value(0.05, scale, tokens_per_call=1800)

    assert v == pytest.approx(0.05 * 1e6 / (0.5 * 1800))


def test_conversion_preserves_the_oracles_ranking():
    """The whole point of the conversion: both rules pick the same model."""
    scale = QualityScale(slope=0.5, intercept=0.5, anchors=["a", "b", "c"])
    tokens, v_oracle = 1800, 0.05
    v_router = router_value_from_oracle_value(v_oracle, scale, tokens)

    # (success probability, USD per 1M blended tokens)
    contenders = [(0.40, 0.20), (0.55, 1.50), (0.70, 9.00)]

    oracle_ranking = sorted(
        contenders,
        key=lambda c: -(c[0] * v_oracle - c[1] * tokens / 1e6),
    )
    router_ranking = sorted(
        contenders,
        key=lambda c: -(scale(c[0]) * v_router - c[1]),
    )

    assert oracle_ranking == router_ranking
