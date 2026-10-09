"""Rasch (item-response) fit over the outcome grid, and the blend that ships.

Why this exists
---------------
``stratum_posterior`` pools trials across a difficulty bucket, which is tight
enough to rank models but throws away two facts the grid plainly contains: that
some models are better *overall*, and that some items are harder *for everyone*.
A cell nobody measured falls back to the prior; a cell with one trial barely
moves off it. Both have already produced wrong conclusions here — sparse premium
coverage made escalation look harmful, and n=1 shrinkage put router accuracy at
0.62 against a measured 0.87.

The Rasch model says the same data more efficiently::

    P(model i solves item j) = sigmoid(ability_i - difficulty_j)

Every observation of a model informs its ability, which then informs every
unmeasured cell for that model; likewise for item difficulty. On the current
grid that is ~3,000 observations estimating ~220 parameters instead of ~3,400
independent ones.

What the cross-validation actually said
---------------------------------------
From ``scripts/fit_irt.py`` (5-fold, held-out log-loss per trial, lower better):

    Rasch + stratum (mean)      0.2253   <- adopted
    2PL + stratum (mean)        0.2257
    per-model x stratum         0.2387   <- current default
    Rasch (IRT) alone           0.2489
    CONTROL stratum + per-model 0.2670
    CONTROL stratum + global    0.2781
    2PL (IRT) alone             0.3333

Three things to read off that table before touching this module:

1. **Rasch alone loses.** It is worse than the stratum estimator already in use.
   Only the blend wins. Do not "simplify" this by dropping the stratum term.
2. **The blend beats both controls.** Averaging two probabilities shrinks
   confident predictions toward the middle, and log-loss rewards that whenever
   the base estimator is overconfident — which the stratum estimator is on small
   per-stratum counts. Blending with an uninformative partner (global mean, or
   the per-model mean) gains 0.011-0.016; blending with Rasch gains 0.013 *more*
   than the better control. That margin is the information; the rest is
   temperature. If a future change makes the blend no better than
   ``CONTROL stratum + per-model``, this module is cargo-culting a temperature
   adjustment and should be deleted.
3. **2PL is not worth it.** Per-item discrimination is strictly worse alone and a
   rounding error in the blend, at triple the parameters. Not implemented here.

The margin over the current default is 0.0134 nats/trial (~5.6% relative). Real,
measured, and small — which is why this is opt-in rather than the default.

Usage
-----
    from theriad.eval.irt import fit_rasch
    rasch = fit_rasch(grid)
    oracle = ExpectedValueOracle(grid, costs, value_correct=0.01,
                                 strata=strata, rasch=rasch)

Passing ``rasch=None`` (the default) reproduces the previous behaviour exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

from .grid import OutcomeGrid

# Fit hyperparameters. These reproduce scripts/fit_irt.py; changing them
# invalidates the cross-validation numbers quoted above.
RIDGE = 1e-2      # keeps abilities finite when a model solves (or fails) everything
LR = 0.5
STEPS = 4000

#: Weight on the Rasch prediction in the blend. 0.5 is what was cross-validated.
#: This is deliberately not a tuned parameter — tuning it on the same folds that
#: selected the estimator would re-introduce the leak that made the first version
#: of fit_irt.py report the wrong winner.
BLEND_WEIGHT = 0.5


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    # Avoid overflow for large negative z.
    e = exp(z)
    return e / (1.0 + e)


@dataclass
class RaschFit:
    """Fitted abilities and difficulties, on the logit scale.

    Difficulties are centred at zero: only ``ability - difficulty`` is
    identified by the data, so the level is arbitrary and fixing it keeps
    successive fits comparable.

    A model or item absent from the fit gets ability/difficulty 0.0, which makes
    ``probability`` return 0.5 — an explicit "no information" answer rather than
    a silent KeyError. Callers that care should check membership first.
    """
    ability: dict[str, float] = field(default_factory=dict)
    difficulty: dict[str, float] = field(default_factory=dict)

    def probability(self, model: str, item_id: str) -> float:
        """P(model solves item) under the fit."""
        return _sigmoid(
            self.ability.get(model, 0.0) - self.difficulty.get(item_id, 0.0)
        )

    def covers(self, model: str, item_id: str) -> bool:
        """True when both sides of the comparison were actually estimated."""
        return model in self.ability and item_id in self.difficulty


def fit_rasch(
    grid: OutcomeGrid,
    ridge: float = RIDGE,
    steps: int = STEPS,
    lr: float = LR,
) -> RaschFit:
    """Maximum a posteriori Rasch fit by gradient ascent.

    Requires numpy, imported lazily so that ``theriad.eval`` stays importable
    without it — only this offline fit needs the dependency.

    Cells with zero trials are skipped: they carry no likelihood, and including
    them would let a model's coverage pattern masquerade as ability.
    """
    import numpy as np

    rows = [
        (cell.model, cell.item_id, cell.successes, cell.trials)
        for cell in grid.cells()
        if cell.trials > 0
    ]
    if not rows:
        return RaschFit()

    models = sorted({r[0] for r in rows})
    items = sorted({r[1] for r in rows})
    mi = {m: i for i, m in enumerate(models)}
    ii = {t: i for i, t in enumerate(items)}

    m_idx = np.array([mi[r[0]] for r in rows], dtype=int)
    i_idx = np.array([ii[r[1]] for r in rows], dtype=int)
    succ = np.array([r[2] for r in rows], dtype=float)
    trials = np.array([r[3] for r in rows], dtype=float)

    n_m, n_i = len(models), len(items)
    ability = np.zeros(n_m)
    difficulty = np.zeros(n_i)
    scale_m = max(1.0, len(rows) / n_m)
    scale_i = max(1.0, len(rows) / n_i)

    for _ in range(steps):
        z = ability[m_idx] - difficulty[i_idx]
        p = 1.0 / (1.0 + np.exp(-z))
        resid = succ - trials * p            # dLL/dz per observation
        g_a = np.bincount(m_idx, weights=resid, minlength=n_m) - ridge * ability
        g_d = -np.bincount(i_idx, weights=resid, minlength=n_i) - ridge * difficulty
        ability += lr * g_a / scale_m
        difficulty += lr * g_d / scale_i
        difficulty -= difficulty.mean()      # identifiability

    return RaschFit(
        ability={m: float(ability[mi[m]]) for m in models},
        difficulty={t: float(difficulty[ii[t]]) for t in items},
    )


def blend(stratum_mean: float, rasch_mean: float, weight: float = BLEND_WEIGHT) -> float:
    """Convex combination of the stratum estimate and the Rasch estimate.

    ``weight`` is the mass on Rasch. See the module docstring for why this is a
    plain average and not something cleverer.
    """
    return (1.0 - weight) * stratum_mean + weight * rasch_mean
