"""Simulate cross-tier escalation before implementing it in the router.

Why simulate first
------------------
Escalation is a production behaviour change on every request. The question is
not "does it escalate?" (any threshold will) but "does escalating reduce
regret?", and that is measurable offline against the existing grid. This script
answers it without touching theriad/models/, so a negative result costs nothing.

What is varied
--------------
1. quality signal — the registry's hand-set ``quality_prior`` (compressed:
   0.042 spread inside premium) vs Rasch ability marginalised over the grid's
   difficulty distribution (0.125 spread, and it reproduces measured accuracy
   to ~0.01).
2. ceiling — how far up the ladder escalation may climb.

The selection rule is the production one, ``quality x V - cost``, with a single
V per *request* (taken from the tier the routing layers chose) rather than one V
per candidate's home tier. Scoring a frontier candidate at V=60 against a
cheap_bulk candidate at V=2 would make frontier win by construction regardless
of quality, which measures nothing.

Run:  python3 scripts/sim_escalation.py
"""
from __future__ import annotations

import numpy as np

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
from theriad.models.unified import blended_cost, value_of_quality

# Escalation ladders, cheapest first. cheap_code/code_strong form the code lane;
# escalating cheap_bulk into a code tier would cross task domains, so the lanes
# are kept separate and only rejoin at premium.
GENERAL_LANE = ["cheap_bulk", "strong_reasoning", "premium"]
CODE_LANE = ["cheap_code", "code_strong", "premium"]
VALUES = [0.001, 0.01, 0.1, 1.0]


def lane_from(tier: str) -> list[str]:
    for lane in (CODE_LANE, GENERAL_LANE):
        if tier in lane:
            return lane[lane.index(tier):]
    return [tier]


def marginal_quality(grid: OutcomeGrid) -> dict[str, float]:
    """P(model solves a random item), integrating the fitted difficulty prior.

    This is the no-per-request-signal case: it is the best the router can do
    while it cannot estimate how hard *this* prompt is.
    """
    fit = fit_rasch(grid)
    difficulties = np.array(list(fit.difficulty.values()))
    return {
        model: float(np.mean(1.0 / (1.0 + np.exp(-(ability - difficulties)))))
        for model, ability in fit.ability.items()
    }


#: Tokens assumed per request when converting between the two V units.
#: ``blended_cost`` is per-million-tokens, so a per-request expected value is
#: ``q * value_correct - blended_cost / 1e6 * TOKENS``. Multiplying through by
#: ``1e6 / TOKENS`` puts it in the router's units, giving
#: ``V(tier) = value_correct * (1e6 / TOKENS)``. At 500 tokens that factor is
#: 2000, so the shipped ladder (cheap_bulk=2.0 ... frontier=60.0) encodes a
#: correct answer being worth $0.001 to $0.03. That is the under-escalation:
#: at those valuations, declining to escalate is the correct decision.
TOKENS_PER_REQUEST = 500
V_SCALE = 1e6 / TOKENS_PER_REQUEST


def build_router(
    *,
    ceiling: str | None,
    quality: dict[str, float] | None,
    v_override: float | None = None,
):
    """Router that may climb its lane up to ``ceiling``.

    ceiling=None reproduces today's behaviour: resolve within the chosen tier
    only. ``quality=None`` scores on the registry prior.
    """
    from theriad.layers.layer0.layer import build_default_layer
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer

    layer0 = build_default_layer()
    layer1 = FeatureRoutingLayer()

    def _q(candidate) -> float:
        if quality is None:
            return candidate.quality_prior or 0.0
        return quality.get(candidate.model, candidate.quality_prior or 0.0)

    def _route(item) -> Action:
        ctx: dict = {}
        result = layer0.route(item.prompt, ctx) or layer1.route(item.prompt, ctx)
        tier = result.decision.tier if result else "cheap_bulk"

        lane = lane_from(tier)
        if ceiling is not None and ceiling in lane:
            lane = lane[: lane.index(ceiling) + 1]
        elif ceiling is None:
            lane = lane[:1]

        # One V for the whole comparison: the request's, not each candidate's.
        v = v_override if v_override is not None else value_of_quality(tier)
        best, best_score = None, -float("inf")
        for rung in lane:
            for candidate in REGISTRY.candidates_for_tier(rung) or []:
                score = _q(candidate) * v - blended_cost(candidate)
                if score > best_score:
                    best, best_score = candidate, score
        return Action(model=best.model, n_samples=1)

    return _route


def main() -> None:
    items = load_corpus("eval_corpus.jsonl")
    grid = OutcomeGrid.load("eval_grid.json")
    strata = derive_strata(grid, items)

    costs = {}
    for tier in REGISTRY.all_tiers():
        for candidate in REGISTRY.candidates_for_tier(tier) or []:
            cost = blended_cost(candidate)
            if cost != float("inf"):
                costs[candidate.model] = cost / 1e6 * 500

    rasch_quality = marginal_quality(grid)

    arms = [
        ("today (no escalation, prior)", None, None),
        ("no escalation, Rasch quality", None, rasch_quality),
        ("escalate->strong_reasoning, prior", "strong_reasoning", None),
        ("escalate->strong_reasoning, Rasch", "strong_reasoning", rasch_quality),
        ("escalate->premium, prior", "premium", None),
        ("escalate->premium, Rasch", "premium", rasch_quality),
    ]

    # The decisive arm: escalate to premium on Rasch quality, with the router's
    # V set to the same value of correctness the harness is scoring against.
    # If the router is told a correct answer is worth what the evaluation says
    # it is worth, does escalation start paying for itself?
    arms.append(("escalate->premium, Rasch, V matched", "premium", rasch_quality))

    header = f"{'arm':36}" + "".join(f"{f'V={v}':>11}" for v in VALUES) + f"{'acc':>8}"
    print(header)
    print("-" * len(header))
    for label, ceiling, quality in arms:
        matched = label.endswith("V matched")
        cells, accuracy = [], 0.0
        for v in VALUES:
            router = build_router(
                ceiling=ceiling,
                quality=quality,
                v_override=v * V_SCALE if matched else None,
            )
            oracle = ExpectedValueOracle(
                grid, costs, value_correct=v, strata=strata
            )
            report = evaluate_router(router, items, oracle, costs, strata=strata)
            summary = report.summary()
            cells.append(summary["regret_per_1k_requests_usd"])
            accuracy = summary["router_accuracy"]
        suffix = "" if matched else f"{accuracy:>8.4f}"
        print(f"{label:36}" + "".join(f"{c:>11.3f}" for c in cells) + suffix)

    from collections import Counter

    for v in VALUES:
        router = build_router(
            ceiling="premium", quality=rasch_quality, v_override=v * V_SCALE
        )
        chosen = Counter(router(item).model for item in items)
        oracle = ExpectedValueOracle(grid, costs, value_correct=v, strata=strata)
        acc = evaluate_router(router, items, oracle, costs, strata=strata).summary()
        print(f"\nV matched at value_correct=${v} (router V={v * V_SCALE:g}) "
              f"acc={acc['router_accuracy']:.4f} cost/1k=${acc['router_cost_usd'] / 200 * 1000:.3f}")
        for model, n in chosen.most_common():
            print(f"   {n:>4}  {model}")


if __name__ == "__main__":
    main()
