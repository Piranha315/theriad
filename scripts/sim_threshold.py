"""Simulate quality-constrained routing: cheapest model clearing a success bar.

Why this formulation
--------------------
``scripts/sim_escalation.py`` showed that maximising ``quality x V - cost``
fixes under-escalation but collapses the router to one model per price point,
and that making V a per-request input would mean asking the caller what a
correct answer is worth — which is the same decision as asking them to pick a
model, and contradicts the product thesis.

So invert it. Do not price quality; constrain it:

    pick the cheapest candidate with P(success | this request) >= tau

``tau`` is one service-level constant Theriad sets once, not an economic input
from the caller. And because P(success) is conditioned on *this* request's
predicted difficulty, an easy prompt clears tau on the free model while a hard
one does not — the selection differentiates by construction, which is exactly
what the V formulation could not do.

    P(model solves item) = sigmoid(rasch_ability[model] - predicted_difficulty)

Predicted difficulty comes from ``scripts/fit_difficulty.py``. This script uses
its **out-of-fold** predictions: routing on an in-sample difficulty would score a
predictor that does not exist at inference time.

Two controls bracket the result:
  - PERFECT difficulty (the true Rasch difficulty) — the ceiling a better
    predictor could reach, not a shippable arm.
  - BLIND (every request assigned the mean difficulty) — what tau alone buys
    without any per-request signal. If the real predictor does not beat BLIND,
    the difficulty model is decorative and only tau is doing work.

Run:  python3 scripts/sim_threshold.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from theriad.eval import (
    ExpectedValueOracle,
    OutcomeGrid,
    derive_strata,
    fit_rasch,
    load_corpus,
)
from theriad.eval.oracle import Action
from theriad.eval.regret import evaluate_router
from theriad.models.registry import DEFAULT_REGISTRY as REGISTRY
from theriad.models.unified import blended_cost

ROOT = Path(__file__).resolve().parent.parent
VALUES = [0.001, 0.01, 0.1, 1.0]
TAUS = [0.50, 0.70, 0.80, 0.90, 0.95]

# Candidate pool: the general lane up to the premium ceiling. Frontier is
# excluded — tiers.py:40 says it is entered deliberately, not as a routine
# escalation target.
LADDER = ["cheap_bulk", "strong_reasoning", "premium"]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def pool() -> list:
    seen, out = set(), []
    for tier in LADDER:
        for candidate in REGISTRY.candidates_for_tier(tier) or []:
            if candidate.model in seen:
                continue
            seen.add(candidate.model)
            out.append(candidate)
    return out


def build_router(candidates, ability, difficulty_of, tau):
    """Cheapest candidate clearing tau; if none clears it, the most capable."""
    ranked = sorted(candidates, key=blended_cost)

    def _route(item) -> Action:
        d = difficulty_of(item)
        for candidate in ranked:
            a = ability.get(candidate.model)
            if a is not None and sigmoid(a - d) >= tau:
                return Action(model=candidate.model, n_samples=1)
        best = max(ranked, key=lambda c: ability.get(c.model, -float("inf")))
        return Action(model=best.model, n_samples=1)

    return _route


def main() -> None:
    items = load_corpus(ROOT / "eval_corpus.jsonl")
    grid = OutcomeGrid.load(ROOT / "eval_grid.json")
    strata = derive_strata(grid, items)
    fit = fit_rasch(grid)

    fitted = json.loads((ROOT / "difficulty_fit.json").read_text())
    oof = fitted["oof_prediction"]
    mean_difficulty = sum(fit.difficulty.values()) / len(fit.difficulty)

    costs, candidates = {}, pool()
    for candidate in candidates:
        cost = blended_cost(candidate)
        if cost != float("inf"):
            costs[candidate.model] = cost / 1e6 * 500

    sources = {
        "PREDICTED": lambda it: oof.get(it.id, mean_difficulty),
        "PERFECT (ceiling)": lambda it: fit.difficulty.get(it.id, mean_difficulty),
        "BLIND (control)": lambda it: mean_difficulty,
    }

    baseline = {}
    for v in VALUES:
        oracle = ExpectedValueOracle(grid, costs, value_correct=v, strata=strata)
        report = evaluate_router(
            lambda it: Action(model="nvidia/nemotron-3-super-120b-a12b:free"),
            items, oracle, costs, strata=strata,
        )
        baseline[v] = report.summary()["regret_per_1k_requests_usd"]

    header = (f"{'difficulty':20}{'tau':>6}"
              + "".join(f"{f'V={v}':>10}" for v in VALUES)
              + f"{'acc':>8}{'models':>8}")
    print(header)
    print("-" * len(header))
    print(f"{'today (no escalation)':20}{'-':>6}"
          + "".join(f"{baseline[v]:>10.3f}" for v in VALUES)
          + f"{0.8690:>8.4f}{1:>8}")
    print("-" * len(header))

    best = None
    for label, difficulty_of in sources.items():
        for tau in TAUS:
            router = build_router(candidates, fit.ability, difficulty_of, tau)
            chosen = Counter(router(it).model for it in items)
            cells, accuracy = [], 0.0
            for v in VALUES:
                oracle = ExpectedValueOracle(
                    grid, costs, value_correct=v, strata=strata
                )
                summary = evaluate_router(
                    router, items, oracle, costs, strata=strata
                ).summary()
                cells.append(summary["regret_per_1k_requests_usd"])
                accuracy = summary["router_accuracy"]
            print(f"{label:20}{tau:>6.2f}"
                  + "".join(f"{c:>10.3f}" for c in cells)
                  + f"{accuracy:>8.4f}{len(chosen):>8}")
            if label == "PREDICTED" and (best is None or cells[2] < best[0]):
                best = (cells[2], tau, chosen, accuracy)
        print()

    regret, tau, chosen, accuracy = best
    print(f"best PREDICTED arm by V=$0.10 regret: tau={tau:.2f} "
          f"regret={regret:.3f} acc={accuracy:.4f}")
    for model, n in chosen.most_common():
        print(f"   {n:>4}  {model}")


if __name__ == "__main__":
    main()
