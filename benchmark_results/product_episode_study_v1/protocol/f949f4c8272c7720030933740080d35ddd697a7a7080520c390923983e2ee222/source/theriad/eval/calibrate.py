"""
Turn a measured outcome grid into router parameters.

This is the step that was missing. The harness could already measure success
rates and report regret, but the numbers the router actually consults —
per-axis quality and per-tier value-of-quality — were hand-authored constants
that somebody had to edit after reading a report. This module computes them and
writes :mod:`theriad.models.calibration`, so re-running the eval re-routes
production.

Two conversions do the real work, and both are places where a plausible-looking
shortcut is wrong.

1. Success rate → quality, by anchoring
---------------------------------------
A measured success rate is not a ``quality_prior``. The registry's priors live
on a 0.68–0.92 band derived from the Artificial Analysis index via
:func:`theriad.models.capabilities.index_to_quality`; a raw success rate on a
hard corpus might be 0.31 for the same model. Writing the raw rate into
``quality_profile`` would not "add a measurement" — it would demote every
measured candidate below every unmeasured one, for reasons that have nothing to
do with capability, because ``quality_for()`` compares profile scores and scalar
priors inside a single ranking.

So the rate is mapped onto the priors' band by an affine fit anchored on the
models that *do* have a published index (``source="artificial_analysis…"``).
Those models appear on both scales, which is what makes the two commensurable.
The fit is refused rather than approximated when there are too few anchors or
the slope comes out non-positive — a non-positive slope means the corpus did not
separate the models at all, and forcing a scale onto that would launder noise
into the router.

Constants shift every candidate equally and cannot change a ranking, so only the
slope matters for routing; the intercept exists to keep the written numbers
legible next to the hand-authored ones.

2. Oracle V → router V, by units
--------------------------------
``theriad.eval.regret`` sweeps *value of a correct answer*, in USD per request.
``theriad.models.unified`` uses *value of quality*, in USD per 1M blended tokens
per unit of quality. These are different quantities and equating them is a units
error that would silently mis-set the most important number in the router.

The conversion falls out of the two ranking rules. The oracle ranks by
``p·V_oracle − cost_per_call``; the router ranks by ``q·V_router − cost_per_1M``
where ``q = a·p + b`` from the fit above. Multiplying the oracle's rule by
``1e6/T`` (T = tokens per call) puts both in per-1M-token units, and the
orderings coincide exactly when::

    V_router = V_oracle × 1e6 / (a × T)

The intercept ``b`` drops out, as it must: it adds the same constant to every
candidate.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from theriad.models.calibration import Calibration
from theriad.models.capabilities import (
    AXIS_BULK,
    AXIS_CODE,
    AXIS_KNOWLEDGE,
    AXIS_REASONING,
)
from theriad.models.registry import DEFAULT_REGISTRY
from theriad.models.tiers import ModelCandidate

from .corpus import EvalItem
from .grid import OutcomeGrid

# ── Configuration ─────────────────────────────────────────────────────────────

AXIS_BY_CATEGORY: dict[str, str] = {
    "code": AXIS_CODE,
    "reasoning": AXIS_REASONING,
    "math": AXIS_REASONING,
    "qa": AXIS_KNOWLEDGE,
    "knowledge": AXIS_KNOWLEDGE,
    "summarize": AXIS_BULK,
    "extract": AXIS_BULK,
    "bulk": AXIS_BULK,
}
"""Corpus category → capability axis. Categories with no mapping are pooled into
the aggregate fit but produce no axis score, because an axis nobody targeted a
benchmark at is exactly the kind of invented number D-051 forbids."""

ANCHOR_SOURCE_PREFIX = "artificial_analysis"
"""Registry ``source`` marking a prior that came from a published index. Only
these can anchor the scale — a hand-authored prior anchored to itself would make
the fit circular."""

MIN_ANCHORS = 3
"""Below this the affine fit is refused. Two points fit any line exactly,
including one drawn through noise."""

MIN_AXIS_TRIALS = 20
"""Fewest graded trials before an axis score is written for a model. Under this
the posterior is wide enough that the published prior is the better estimate."""

SHRINKAGE_TRIALS = 60
"""Trial count at which a measurement and the prior carry equal weight. Chosen
so a full 5-sample sweep of a ~50-item axis (250 trials) lands at ~80% measured,
while a thin 25-trial probe stays prior-dominated."""


# ── Measurement ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisMeasurement:
    """Pooled outcomes for one (model, axis)."""

    model: str
    axis: str | None
    trials: int
    successes: int

    @property
    def rate(self) -> float:
        """Jeffreys-smoothed success rate.

        Smoothed rather than raw so a model that swept a small axis does not
        record exactly 1.0 and then get scaled into a quality above every
        frontier model on the strength of twenty questions.
        """
        return (self.successes + 0.5) / (self.trials + 1.0)

    @property
    def weight(self) -> float:
        """Share of the final score taken from measurement rather than prior."""
        return self.trials / (self.trials + SHRINKAGE_TRIALS)


def measure_axes(
    grid: OutcomeGrid, items: list[EvalItem]
) -> dict[str, dict[str | None, AxisMeasurement]]:
    """Pool grid cells by (model, axis). ``None`` is the all-items aggregate."""
    axis_of = {i.id: AXIS_BY_CATEGORY.get(i.category) for i in items}
    totals: dict[tuple[str, str | None], list[int]] = {}

    for model in grid.models():
        for item_id, axis in axis_of.items():
            cell = grid.cell(model, item_id)
            if cell is None or cell.trials == 0:
                continue
            keys: list[tuple[str, str | None]] = [(model, None)]
            if axis is not None:
                keys.append((model, axis))
            for key in keys:
                acc = totals.setdefault(key, [0, 0])
                acc[0] += cell.trials
                acc[1] += cell.successes

    out: dict[str, dict[str | None, AxisMeasurement]] = {}
    for (model, axis), (trials, successes) in totals.items():
        out.setdefault(model, {})[axis] = AxisMeasurement(
            model=model, axis=axis, trials=trials, successes=successes
        )
    return out


# ── Scale fitting ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityScale:
    """Affine map from measured success rate to the registry's quality band."""

    slope: float
    intercept: float
    anchors: list[str]

    def __call__(self, rate: float) -> float:
        return max(0.0, min(1.0, self.slope * rate + self.intercept))


class ScaleFitError(RuntimeError):
    """The measurement cannot be put on the registry's scale."""


def anchor_candidates(
    registry=DEFAULT_REGISTRY,
) -> dict[str, ModelCandidate]:
    """Models whose ``quality_prior`` came from a published index."""
    found: dict[str, ModelCandidate] = {}
    for tier in registry.all_tiers():
        for c in registry.candidates_for_tier(tier):
            if (
                c.quality_prior is not None
                and (c.source or "").startswith(ANCHOR_SOURCE_PREFIX)
            ):
                found.setdefault(c.model, c)
    return found


def fit_quality_scale(
    measurements: dict[str, dict[str | None, AxisMeasurement]],
    registry=DEFAULT_REGISTRY,
) -> QualityScale:
    """Least-squares fit of published prior against measured aggregate rate.

    Raises:
        ScaleFitError: too few anchors, no spread in the measured rates, or a
            non-positive slope. Each means the grid cannot say which model is
            better on a scale comparable to the registry's, and the honest
            outcome is to write no profiles rather than a fabricated ordering.
    """
    anchors = anchor_candidates(registry)
    used: list[str] = []
    points: list[tuple[float, float]] = []
    for model, cand in sorted(anchors.items()):
        m = measurements.get(model, {}).get(None)
        if m is None or m.trials < MIN_AXIS_TRIALS:
            continue
        used.append(model)
        points.append((m.rate, cand.quality_prior))
    if len(points) < MIN_ANCHORS:
        raise ScaleFitError(
            f"{len(points)} usable anchors, need {MIN_ANCHORS}. Anchors are "
            f"registry models with source='{ANCHOR_SOURCE_PREFIX}…' and at least "
            f"{MIN_AXIS_TRIALS} trials in the grid; extend coverage to more of "
            f"them: {sorted(anchors)}"
        )

    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx <= 1e-12:
        raise ScaleFitError(
            "anchor models all scored the same measured rate; the corpus does "
            "not separate them, so no scale can be fitted"
        )
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    if slope <= 0.0:
        raise ScaleFitError(
            f"fitted slope {slope:.4f} is not positive: on this corpus the "
            "models with higher published indices did not score higher. Treat "
            "that as a finding about the corpus (contamination, grading, or "
            "truncation), not as a scale to route on."
        )
    return QualityScale(slope=slope, intercept=mean_y - slope * mean_x, anchors=used)


# ── Quality profiles ──────────────────────────────────────────────────────────


def registry_priors(registry=DEFAULT_REGISTRY) -> dict[str, float]:
    priors: dict[str, float] = {}
    for tier in registry.all_tiers():
        for c in registry.candidates_for_tier(tier):
            if c.quality_prior is not None:
                priors.setdefault(c.model, c.quality_prior)
    return priors


def quality_profiles(
    measurements: dict[str, dict[str | None, AxisMeasurement]],
    scale: QualityScale,
    registry=DEFAULT_REGISTRY,
) -> dict[str, dict[str, float]]:
    """Measured per-axis quality, shrunk toward each model's existing prior.

    Shrinkage is what keeps a thin measurement from overturning a published
    index. A model with 250 trials on an axis lands ~80% on the measurement; one
    with 25 sits mostly on its prior and moves as coverage grows.
    """
    priors = registry_priors(registry)
    profiles: dict[str, dict[str, float]] = {}
    for model, by_axis in sorted(measurements.items()):
        prior = priors.get(model)
        axes: dict[str, float] = {}
        for axis, m in sorted(by_axis.items(), key=lambda kv: (kv[0] is None, kv[0])):
            if axis is None or m.trials < MIN_AXIS_TRIALS:
                continue
            measured = scale(m.rate)
            axes[axis] = round(
                measured if prior is None
                else m.weight * measured + (1 - m.weight) * prior,
                4,
            )
        if axes:
            profiles[model] = axes
    return profiles


# ── Value of quality ──────────────────────────────────────────────────────────

DEFAULT_TOKENS_PER_CALL = 1700
"""Input + output tokens assumed per request when converting oracle V to router
V. Must match the assumption the grid's cost estimates were built with
(``DEFAULT_INPUT_TOKENS + DEFAULT_OUTPUT_TOKENS`` in theriad.eval.run)."""


def router_value_from_oracle_value(
    v_oracle: float, scale: QualityScale, tokens_per_call: int = DEFAULT_TOKENS_PER_CALL
) -> float:
    """Convert USD-per-correct-answer into the router's USD-per-1M-tokens V.

    See the module docstring for the derivation. The intercept does not appear
    because it shifts every candidate equally.
    """
    if tokens_per_call <= 0:
        raise ValueError("tokens_per_call must be positive")
    return v_oracle * 1e6 / (scale.slope * tokens_per_call)


def best_oracle_value(sweep: list[dict]) -> tuple[float, bool] | None:
    """``(V, interior)`` — the swept V with least regret, and whether it is identified.

    ``interior`` is False when the minimum sits on an endpoint of the swept
    range. That is not an optimum, it is the edge of the search: regret was
    still falling when the sweep ran out, so the number reported is an artifact
    of how the range was chosen. Callers must not install a boundary hit —
    widen ``--values`` and re-run instead.

    Ties break toward the lower V: when two values are indistinguishable the
    cheaper one is the one that does not spend on an unproven difference.
    """
    scored = [
        (row["summary"]["regret_per_1k_requests_usd"], row["value_correct"])
        for row in sweep
        if "summary" in row
    ]
    if not scored:
        return None
    best = min(scored)[1]
    swept = sorted(v for _, v in scored)
    interior = len(swept) > 2 and swept[0] < best < swept[-1]
    return best, interior


def tier_models(tier: str, registry=DEFAULT_REGISTRY) -> list[str]:
    return [c.model for c in registry.candidates_for_tier(tier)]


def calibrate_value_of_quality(
    grid: OutcomeGrid,
    items: list[EvalItem],
    router,
    scale: QualityScale,
    values: list[float],
    tiers: list[str] | None = None,
    registry=DEFAULT_REGISTRY,
    tokens_per_call: int = DEFAULT_TOKENS_PER_CALL,
) -> tuple[dict[str, float], dict[str, str]]:
    """``(measured V per tier, reasons tiers were skipped)``.

    A tier is skipped, not defaulted, whenever the sweep cannot identify its V —
    too few measured candidates, or an optimum on the edge of the swept range.
    Those tiers keep their hand-authored value, which is at least an honest
    guess, rather than a number laundered through an inconclusive sweep.
    """
    from .run import sweep_value  # local import: run imports this module's CLI

    measured = set(grid.models())
    out: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for tier in tiers or registry.all_tiers():
        models = [m for m in tier_models(tier, registry) if m in measured]
        if len(models) < 2:
            skipped[tier] = f"only {len(models)} of its candidates are in the grid"
            continue
        result = best_oracle_value(sweep_value(grid, items, router, values, models))
        if result is None:
            skipped[tier] = "sweep produced no scored rows"
            continue
        v_oracle, interior = result
        if not interior:
            skipped[tier] = (
                f"least regret at V={v_oracle:g}, an endpoint of the swept range "
                f"— widen --values past it and re-run"
            )
            continue
        out[tier] = round(
            router_value_from_oracle_value(v_oracle, scale, tokens_per_call), 4
        )
    return out, skipped


# ── Entry point ───────────────────────────────────────────────────────────────


def build_calibration(
    grid: OutcomeGrid,
    items: list[EvalItem],
    router,
    values: list[float],
    registry=DEFAULT_REGISTRY,
    tokens_per_call: int = DEFAULT_TOKENS_PER_CALL,
) -> tuple[Calibration, QualityScale, dict[str, str]]:
    """``(calibration, scale, skipped tiers with reasons)``.

    Raises:
        ScaleFitError: propagated from :func:`fit_quality_scale`. Refusing here
            is deliberate — a calibration written on an unfittable grid would be
            indistinguishable, at the router, from one written on a good grid.
    """
    measurements = measure_axes(grid, items)
    scale = fit_quality_scale(measurements, registry)
    value_of_quality, skipped = calibrate_value_of_quality(
        grid, items, router, scale, values,
        registry=registry, tokens_per_call=tokens_per_call,
    )
    calibration = Calibration(
        quality_profiles=quality_profiles(measurements, scale, registry),
        value_of_quality=value_of_quality,
        grid_trials=grid.total_trials,
        generated_at=_dt.date.today().isoformat(),
        corpus_sources=sorted({i.source for i in items}),
    )
    return calibration, scale, skipped
