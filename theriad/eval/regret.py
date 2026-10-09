"""
Router regret: what the routing policy left on the table versus the oracle.

Regret is reported in expected-value units (USD), which makes it comparable
across corpora and directly interpretable: "this policy gives up $X per thousand
requests against a perfectly-informed router at this value-of-correctness".

It decomposes into two parts that call for different fixes:

- **overspend** — the router chose a more expensive action than the oracle for no
  compensating accuracy. Fix by lowering escalation thresholds.
- **accuracy loss** — the router chose too cheap an action and gave up success
  probability worth more than it saved. Fix by raising them.

A single scalar hides which of those is happening, and they push the tuning knob
in opposite directions, so both are always reported.

Breakdowns by difficulty stratum and by corpus vintage are the point of the
harness rather than extras. Flat regret across vintages suggests the router's
decision boundary transfers; regret climbing on recent items suggests it is
fitting contamination in the older ones (see theriad/eval/corpus.py).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .corpus import EvalItem
from .grid import Strata
from .oracle import Action, ExpectedValueOracle, ScoredAction, score_action

# A router under test: given an item, return the action it would take.
RouterFn = Callable[[EvalItem], Action]


@dataclass
class ItemRegret:
    """Per-item comparison of router action against the oracle's."""
    item_id: str
    stratum: str
    vintage: int | None
    router: ScoredAction
    oracle: ScoredAction

    @property
    def regret(self) -> float:
        """Oracle EV minus router EV; ≥ 0 up to floating-point error."""
        return self.oracle.expected_value - self.router.expected_value

    @property
    def cost_delta(self) -> float:
        """Router spend minus oracle spend. Positive = overspent."""
        return self.router.cost_usd - self.oracle.cost_usd

    @property
    def accuracy_delta(self) -> float:
        """Router success probability minus oracle's. Negative = gave up accuracy."""
        return self.router.success_prob - self.oracle.success_prob

    @property
    def matched(self) -> bool:
        return self.router.label == self.oracle.label


@dataclass
class RegretReport:
    """Aggregate regret across a corpus."""
    items: list[ItemRegret] = field(default_factory=list)
    value_correct: float = 0.0

    # ── Totals ────────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def total_regret(self) -> float:
        return sum(i.regret for i in self.items)

    @property
    def mean_regret(self) -> float:
        return self.total_regret / self.n if self.n else 0.0

    @property
    def regret_per_1k(self) -> float:
        """Expected value given up per 1,000 requests — the headline number."""
        return self.mean_regret * 1000

    @property
    def match_rate(self) -> float:
        """Fraction of items where the router picked the oracle's action."""
        return sum(1 for i in self.items if i.matched) / self.n if self.n else 0.0

    @property
    def router_cost(self) -> float:
        return sum(i.router.cost_usd for i in self.items)

    @property
    def oracle_cost(self) -> float:
        return sum(i.oracle.cost_usd for i in self.items)

    @property
    def router_accuracy(self) -> float:
        """Mean expected success probability under the router's actions."""
        return (
            sum(i.router.success_prob for i in self.items) / self.n if self.n else 0.0
        )

    @property
    def oracle_accuracy(self) -> float:
        return (
            sum(i.oracle.success_prob for i in self.items) / self.n if self.n else 0.0
        )

    # ── Decomposition ─────────────────────────────────────────────────────────

    @property
    def overspend(self) -> float:
        """Regret attributable to spending more than the oracle."""
        return sum(i.cost_delta for i in self.items if i.cost_delta > 0)

    @property
    def accuracy_loss_value(self) -> float:
        """Regret attributable to giving up success probability, in USD."""
        return sum(
            -i.accuracy_delta * self.value_correct
            for i in self.items
            if i.accuracy_delta < 0
        )

    # ── Breakdowns ────────────────────────────────────────────────────────────

    def by_stratum(self) -> dict[str, "RegretReport"]:
        return self._group(lambda i: i.stratum)

    def by_vintage(self) -> dict[int | None, "RegretReport"]:
        return self._group(lambda i: i.vintage)

    def _group(self, key) -> dict:
        buckets: dict = {}
        for item in self.items:
            buckets.setdefault(key(item), []).append(item)
        return {
            k: RegretReport(items=v, value_correct=self.value_correct)
            for k, v in sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0]))
        }

    def summary(self) -> dict:
        return {
            "n_items": self.n,
            "value_correct_usd": self.value_correct,
            "regret_per_1k_requests_usd": round(self.regret_per_1k, 4),
            "match_rate": round(self.match_rate, 4),
            "router_cost_usd": round(self.router_cost, 6),
            "oracle_cost_usd": round(self.oracle_cost, 6),
            "router_accuracy": round(self.router_accuracy, 4),
            "oracle_accuracy": round(self.oracle_accuracy, 4),
            "overspend_usd": round(self.overspend, 6),
            "accuracy_loss_usd": round(self.accuracy_loss_value, 6),
        }


def evaluate_router(
    router: RouterFn,
    items: list[EvalItem],
    oracle: ExpectedValueOracle,
    costs: dict[str, float],
    strata: Strata | None = None,
    models: list[str] | None = None,
) -> RegretReport:
    """Score ``router`` against ``oracle`` over ``items``.

    The router's own action is scored with the same probability estimates and
    cost model as the oracle's, so regret isolates the *decision* rather than
    disagreement about the inputs.
    """
    report = RegretReport(value_correct=oracle._value)  # noqa: SLF001 - same package
    for item in items:
        choice = oracle.evaluate(item, models)
        chosen = router(item)
        scored_router = choice.find(chosen.label) or score_action(
            chosen,
            oracle.success_probability(chosen.model, item),
            costs.get(chosen.model, 0.0),
            oracle._value,  # noqa: SLF001 - same package
        )
        report.items.append(
            ItemRegret(
                item_id=item.id,
                stratum=strata.stratum(item.id) if strata else "all",
                vintage=item.vintage,
                router=scored_router,
                oracle=choice.best,
            )
        )
    return report
