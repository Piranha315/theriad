"""
Expected-value oracle over routing actions.

The oracle answers "what was the best action available for this item", so the
router can be scored against it. It is deliberately *not* "the cheapest model
that solved it": that formulation reads a Bernoulli draw as a property of the
item, and at feasible sample sizes the draw is mostly noise.

Why expected value rather than "did it solve it"
------------------------------------------------
No model has a success probability of exactly zero on a given item — outputs are
stochastic, and resampling changes answers — so "no model solves this" is a
statement about the sampling budget. The decision is therefore not "is p > 0"
but whether the *gap* justifies the spend:

    escalate  ⟺  Δp × V  >  Δcost

where V is the value of a correct answer. V has to be stated, not assumed:
"is 5x the price worth +10 points of accuracy" has no answer until someone says
what an answer is worth. Making it an explicit parameter keeps that judgement
visible in the report instead of buried in a metric definition.

Estimating p
------------
Success rates come from a Beta-Binomial posterior with a uniform Beta(1, 1)
prior, so a cell with zero observed successes yields a small positive estimate
rather than collapsing to 0.0. Posterior mean is (successes + 1) / (trials + 2).

Per-item estimates are wide at any affordable k — 1 success in 10 trials gives a
posterior mean near 0.17 with a 95% interval of roughly [0.02, 0.41], nowhere
near tight enough to rank two models on that item alone. So decision-facing
estimates should be pooled per stratum via ``stratum_posterior``. That is also
the right resolution on principle: the router cannot see item identity at
inference time either, so an oracle with per-item knowledge would report a bound
no router could reach.

Best-of-n, and why it is off by default
---------------------------------------
Resampling a cheap model can beat escalating: with per-sample success p, n
samples succeed with probability 1 − (1 − p)^n, so at p=0.08 five samples reach
~34%, beating a single call at p=0.20 for the same spend when the strong model
costs 5x.

That formula assumes samples are **independent**, and they are badly not. A model
that does not know a fact produces five variations of the same wrong answer.
Taking independence at face value made the oracle choose best-of-5 on the
cheapest model for essentially every item — a bound no single-call router can
ever match, which collapsed match rate to zero and made regret measure the
harness's own optimism rather than the router's decisions.

Two corrections follow, and both matter:

- ``max_samples`` defaults to **1**, so the oracle answers "which model" — the
  question the router actually faces. Raise it only when a verifier exists at
  inference time to pick the winning sample.
- When best-of-n is enabled, gains are discounted by
  ``sample_correlation``, which :func:`estimate_sample_correlation` can measure
  from the grid's own over-dispersion rather than assume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .corpus import EvalItem
from .grid import OutcomeGrid, Strata
from .irt import RaschFit, blend

# Uniform Beta(1, 1) prior — Laplace smoothing.
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


@dataclass(frozen=True)
class Posterior:
    """Beta posterior over a success probability."""
    successes: int
    trials: int
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA

    @property
    def a(self) -> float:
        return self.successes + self.alpha

    @property
    def b(self) -> float:
        return (self.trials - self.successes) + self.beta

    @property
    def mean(self) -> float:
        """Posterior mean; never exactly 0 or 1."""
        return self.a / (self.a + self.b)

    @property
    def stdev(self) -> float:
        """Closed-form Beta standard deviation."""
        a, b = self.a, self.b
        total = a + b
        return math.sqrt((a * b) / (total * total * (total + 1.0)))

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        """Normal-approximation credible interval, clipped to [0, 1].

        Approximate by construction: the Beta is skewed near the boundaries,
        which is exactly where the hard tail lives. Treat the width as an
        order-of-magnitude signal about whether k is sufficient, not as an exact
        coverage guarantee.
        """
        m, s = self.mean, self.stdev
        return (max(0.0, m - z * s), min(1.0, m + z * s))


def posterior(grid: OutcomeGrid, model: str, item_id: str) -> Posterior:
    """Per-item posterior for one cell. Wide at realistic k — see module docstring."""
    cell = grid.cell(model, item_id)
    if cell is None:
        return Posterior(successes=0, trials=0)
    return Posterior(successes=cell.successes, trials=cell.trials)


def stratum_posterior(
    grid: OutcomeGrid, model: str, item_ids: list[str]
) -> Posterior:
    """Pooled posterior for one model across a set of items.

    This is the decision-facing estimator: it pools trials across a stratum so
    the estimate is tight enough to rank models, and it matches the information
    the router actually has at inference time.
    """
    successes = 0
    trials = 0
    for item_id in item_ids:
        cell = grid.cell(model, item_id)
        if cell is not None:
            successes += cell.successes
            trials += cell.trials
    return Posterior(successes=successes, trials=trials)


# ── Actions ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Action:
    """A candidate routing action: call ``model`` ``n_samples`` times."""
    model: str
    n_samples: int = 1

    @property
    def label(self) -> str:
        return self.model if self.n_samples == 1 else f"{self.model}×{self.n_samples}"


@dataclass(frozen=True)
class ScoredAction:
    action: Action
    success_prob: float
    cost_usd: float
    expected_value: float

    @property
    def label(self) -> str:
        return self.action.label


DEFAULT_SAMPLE_CORRELATION = 0.5
"""Assumed correlation between repeated samples, when not measured.

``1 - (1-p)^n`` assumes samples are *independent*. They are not: a model that
does not know a fact returns five variations of the same wrong answer, and the
independence formula turns that into a large fictional probability gain. Left
uncorrected, the oracle picks best-of-n on the cheapest model for essentially
every item, no single-call router can ever match it, and regret stops measuring
routing at all.

Correlation is measurable from the grid — see :func:`estimate_sample_correlation`
— and the measured value should be preferred. 0.5 is a deliberately conservative
placeholder for a cold grid, not a finding.
"""


def best_of_n_probability(
    p: float, n: int, correlation: float = 0.0
) -> float:
    """Probability that at least one of n samples succeeds.

    ``correlation`` interpolates between independent samples (0.0, the textbook
    ``1-(1-p)^n``) and perfectly redundant ones (1.0, where n samples are worth
    exactly one). Real repeated sampling sits between: some genuine diversity,
    heavily correlated by what the model does and does not know.
    """
    if n <= 1:
        return p
    independent = 1.0 - (1.0 - p) ** n
    rho = min(max(correlation, 0.0), 1.0)
    return p + (1.0 - rho) * (independent - p)


def estimate_sample_correlation(grid: "OutcomeGrid", model: str) -> float | None:
    """Estimate inter-sample correlation for ``model`` from grid over-dispersion.

    If repeated samples were independent, per-cell success counts would follow
    Binomial(k, p) and the variance of per-cell rates would be ``p(1-p)/k``.
    Perfectly correlated samples make every cell all-or-nothing, with variance
    ``p(1-p)``. The observed variance lands between, and its position is the
    intraclass correlation:

        rho = (k · Var_obs / (p(1-p)) − 1) / (k − 1)

    Returns None when there is too little data to estimate (fewer than two
    multi-trial cells, or a degenerate pooled rate).
    """
    cells = [
        c
        for c in grid._cells.values()  # noqa: SLF001 - same package
        if c.model == model and c.trials > 1
    ]
    if len(cells) < 2:
        return None
    k = sum(c.trials for c in cells) / len(cells)
    p = sum(c.successes for c in cells) / sum(c.trials for c in cells)
    if p <= 0.0 or p >= 1.0 or k <= 1:
        return None
    rates = [c.successes / c.trials for c in cells]
    mean_rate = sum(rates) / len(rates)
    var_obs = sum((r - mean_rate) ** 2 for r in rates) / (len(rates) - 1)
    rho = (k * var_obs / (p * (1.0 - p)) - 1.0) / (k - 1.0)
    return min(max(rho, 0.0), 1.0)


def score_action(
    action: Action,
    p_single: float,
    cost_per_call: float,
    value_correct: float,
    correlation: float = 0.0,
) -> ScoredAction:
    """Score one action's expected value: p × V − cost."""
    p = best_of_n_probability(p_single, action.n_samples, correlation)
    cost = cost_per_call * action.n_samples
    return ScoredAction(
        action=action,
        success_prob=p,
        cost_usd=cost,
        expected_value=p * value_correct - cost,
    )


def enumerate_actions(
    models: list[str], max_samples: int, verifiable: bool
) -> list[Action]:
    """All actions available for an item.

    Best-of-n is offered only when a verifier exists: without one you cannot
    identify the winning sample, so the extra calls buy nothing.
    """
    actions = [Action(model=m, n_samples=1) for m in models]
    if verifiable and max_samples > 1:
        for m in models:
            for n in range(2, max_samples + 1):
                actions.append(Action(model=m, n_samples=n))
    return actions


# ── Oracle ────────────────────────────────────────────────────────────────────


@dataclass
class OracleChoice:
    """The best available action for one item, plus the full ranking."""
    item_id: str
    best: ScoredAction
    all_actions: list[ScoredAction]

    def find(self, label: str) -> ScoredAction | None:
        return next((a for a in self.all_actions if a.label == label), None)


class ExpectedValueOracle:
    """Ranks routing actions by expected value for a stated value-of-correctness.

    Args:
        grid:          Measured outcomes.
        costs:         {model: USD per call} for this corpus's token profile.
        value_correct: V — USD value of a correct answer. The single most
                       consequential parameter: as V rises the oracle tolerates
                       more spend per point of accuracy, so report results as a
                       sweep over V rather than at one arbitrary setting.
        max_samples:   Largest n considered for best-of-n. **Defaults to 1**,
                       making this a pure model-choice oracle — the question the
                       router actually faces. Raise it only when a verifier
                       exists at inference time; otherwise the oracle is scored
                       against a bound no router can reach, and every router
                       looks equally terrible.
        strata:        When provided, success probabilities are pooled per
                       stratum instead of per item — the recommended mode.
        sample_correlation: How redundant repeated samples are, in [0, 1]. Only
                       matters when max_samples > 1. Prefer a value measured
                       from the grid via estimate_sample_correlation.
        rasch:         Optional fitted Rasch model from ``theriad.eval.irt``.
                       When supplied alongside ``strata``, success probabilities
                       become the 50/50 blend of the two, which measured best in
                       cross-validation. Omitted by default: the margin over
                       strata alone is real but small (~5.6% relative), so it is
                       an explicit opt-in rather than a silent change to every
                       existing report.
    """

    def __init__(
        self,
        grid: OutcomeGrid,
        costs: dict[str, float],
        value_correct: float,
        max_samples: int = 1,
        strata: Strata | None = None,
        sample_correlation: float = DEFAULT_SAMPLE_CORRELATION,
        rasch: RaschFit | None = None,
    ):
        self._grid = grid
        self._costs = costs
        self._value = value_correct
        self._max_samples = max_samples
        self._correlation = sample_correlation
        self._strata = strata
        self._rasch = rasch
        self._stratum_cache: dict[tuple[str, str], Posterior] = {}
        self._stratum_members: dict[str, list[str]] = {}
        if strata is not None:
            for item_id, stratum in strata.by_item.items():
                self._stratum_members.setdefault(stratum, []).append(item_id)

    def success_probability(self, model: str, item: EvalItem) -> float:
        """Estimated single-call success probability for model on item.

        With ``rasch`` supplied *and* strata in use, this returns the blend that
        cross-validated best (0.2253 nats/trial vs 0.2387 for strata alone; see
        ``theriad.eval.irt``). The blend needs both halves — Rasch alone scored
        0.2489, worse than strata — so it is only applied when both are present.
        """
        if self._strata is not None:
            stratum = self._strata.stratum(item.id)
            key = (model, stratum)
            if key not in self._stratum_cache:
                members = self._stratum_members.get(stratum, [item.id])
                self._stratum_cache[key] = stratum_posterior(
                    self._grid, model, members
                )
            stratum_mean = self._stratum_cache[key].mean
            if self._rasch is not None and self._rasch.covers(model, item.id):
                return blend(stratum_mean, self._rasch.probability(model, item.id))
            return stratum_mean
        return posterior(self._grid, model, item.id).mean

    def evaluate(self, item: EvalItem, models: list[str] | None = None) -> OracleChoice:
        """Score every available action for ``item`` and return the ranking."""
        candidate_models = models or sorted(self._costs)
        actions = enumerate_actions(
            candidate_models, self._max_samples, item.verifiable
        )
        scored = [
            score_action(
                action,
                self.success_probability(action.model, item),
                self._costs.get(action.model, 0.0),
                self._value,
                self._correlation,
            )
            for action in actions
        ]
        scored.sort(key=lambda s: s.expected_value, reverse=True)
        return OracleChoice(item_id=item.id, best=scored[0], all_actions=scored)
