"""
The unified router: one selection rule, no dials.

Theriad's premise is that a user should not have to choose between "cheap" and
"good". The product's own description — *route prompts to the cheapest model that
can handle them* — already states the rule, and three named policies
(``aggressive_savings`` / ``balanced`` / ``quality_first``) were three different
answers to a question the user should never have been asked. Worse, they
disagreed: the same prompt could land on a $0.14 model or a $2.00 one depending
on a flag most callers left at its default.

The rule
--------
Maximise expected value::

    score = quality x VALUE_OF_QUALITY - blended_cost

The highest score wins. That is "cheapest model that can handle it" made
precise: quality is worth paying for at a stated exchange rate, and beyond that
rate it is not.

Why expected value rather than a quality band
----------------------------------------------
The first version of this module took the cheapest candidate within a fixed
quality tolerance of the best available. That looks reasonable and is quietly
broken, because an absolute quality band ignores the price ratio. On the premium
tier it selected Grok 4.5 at $8.60/1k over DeepSeek V4 Flash 0731 at $0.48/1k —
**18x the price for 0.042 of quality** — purely because 0.042 exceeded the band.
The same band that is too tight there is too loose elsewhere.

Expected value has no such failure mode: it compares the quality *gain* against
the cost *of that gain*, so a large price ratio needs a large quality ratio to
justify it, at every point on the scale.

This is deliberately the same rule the eval harness's oracle uses
(``theriad.eval.oracle.ExpectedValueOracle``). The router and the thing that
grades the router now optimise the same objective, so regret measures the
router's *information* rather than a disagreement about goals.

The one option that remains
---------------------------
:class:`RoutingConstraints` carries ``exclude_origins``. That is the single
user-facing choice Theriad exposes, because it is the one thing the router
genuinely cannot decide on the user's behalf: whether a vendor's country of
origin is acceptable is a policy question about their business, not a
price/quality tradeoff. Everything else is an optimization the router should
just do.

Candidates with no recorded origin are treated as **non-compliant** when a
constraint is active. A model we cannot attribute must not silently satisfy a
restriction someone set deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import load_calibration
from .tiers import ModelCandidate

VALUE_OF_QUALITY = 12.0
"""Exchange rate between quality and price, in USD per 1M blended tokens per
unit of quality.

This is the one number that sets Theriad's character, and it is **ours to set,
not the user's** — that is the whole point of removing the policy dial. At 12,
one hundredth of quality is worth $0.12 per 1M blended tokens, so:

  - a free model loses to a paid one only if the paid one is meaningfully better;
  - the premium tier prefers DeepSeek 0731 (0.834, $0.175 blended) over Grok
    (0.876, $3.00) because +0.042 quality is worth ~$0.50, not $2.83;
  - the frontier tier currently reaches GLM 5.3 Flash: its provisional 0.908
    prior and $0.2375 blended seed price beat Muse Spark 1.2's 0.892 and $2.00
    under this unmeasured default. The replacement-study result still needs to
    validate that assumption; Muse remains the U.S.-only fallback and Opus is an
    explicit comparison control.

Calibrated against the current registry so the selections above hold. The eval
grid supersedes it: once measured success rates exist, V is exactly the
parameter the regret sweep estimates (see theriad/eval/regret.py), and this
default should be replaced with the value that minimises measured regret."""


TIER_VALUE_OF_QUALITY: dict[str, float] = {
    "free_or_floor":      0.0,
    "free_moe_general":   0.0,
    "free_moe_code":      0.0,
    "free_moe_verify":    0.0,
    "free_moe_summarize": 0.0,
    "cheap_bulk":         2.0,
    "cheap_code":         4.0,
    "code_strong":       14.0,
    "strong_reasoning":  14.0,
    "premium":           12.0,
    "frontier":          60.0,
}
"""Per-tier exchange rate. A single global V is wrong because the value of a
correct answer genuinely differs by task: a slightly worse summary costs almost
nothing, a slightly worse architectural decision can cost a great deal.

Low on the bulk tiers, so free models win there unless a paid one is *much*
better — which is what makes "swap the cheap tiers to free" a consequence of the
rule rather than a special case bolted onto it. Zero on the free tiers, where
cost is zero for every candidate and quality is the only remaining term.

These are judgement calls, and they are the highest-leverage numbers in the
router. They are exactly what the eval harness's V sweep measures
(theriad/eval/regret.py): run the grid, find the V that minimises regret per
tier, replace these."""


def value_of_quality(tier: str | None) -> float:
    """Per-tier V: measured where the eval grid has spoken, hand-set otherwise.

    Resolution order is measured → hand-authored → global default. The
    calibration file (``theriad/models/calibration.json``, written by
    ``python -m theriad.eval.run calibrate --write``) is absent on a fresh
    checkout, so this reduces to the table above until a grid exists.
    """
    if tier is None:
        return VALUE_OF_QUALITY
    measured = load_calibration().value_of_quality.get(tier)
    if measured is not None:
        return measured
    return TIER_VALUE_OF_QUALITY.get(tier, VALUE_OF_QUALITY)


@dataclass(frozen=True)
class RoutingConstraints:
    """Hard filters applied before selection.

    Unlike the policies this replaces, these do not express a preference the
    router could reasonably resolve itself — they express a requirement.
    """

    exclude_origins: frozenset[str] = field(default_factory=frozenset)
    """Vendor origins to exclude, e.g. ``frozenset({"CN"})``."""

    allowed_origins: frozenset[str] = field(default_factory=frozenset)
    """When set, only these vendor origins are eligible, e.g. ``{"US"}``."""

    @classmethod
    def none(cls) -> "RoutingConstraints":
        return cls()

    @classmethod
    def exclude_china(cls) -> "RoutingConstraints":
        """The shipped preset: route only to non-China vendors."""
        return cls(exclude_origins=frozenset({"CN"}))

    @classmethod
    def us_only(cls) -> "RoutingConstraints":
        """Route only to candidates whose cataloged vendor origin is US."""
        return cls(allowed_origins=frozenset({"US"}))

    @property
    def active(self) -> bool:
        return bool(self.exclude_origins or self.allowed_origins)

    def permits(self, candidate: ModelCandidate) -> bool:
        if not self.active:
            return True
        # Unknown origin fails a constraint that was set deliberately.
        if not candidate.origin:
            return False
        origin = candidate.origin.upper()
        if self.allowed_origins and origin not in self.allowed_origins:
            return False
        return origin not in self.exclude_origins


NO_CONSTRAINTS = RoutingConstraints()


def blended_cost(candidate: ModelCandidate) -> float:
    """Cost proxy used for ranking: 75% input, 25% output.

    Weighted toward input because routing decisions are made before the response
    exists, and input length is the only side known at that point. Unpriced
    candidates sort last rather than free — an unknown price is a risk, not a
    bargain.
    """
    if candidate.input_cost_per_million is None or candidate.output_cost_per_million is None:
        return float("inf")
    return (
        0.75 * candidate.input_cost_per_million
        + 0.25 * candidate.output_cost_per_million
    )


def select(
    candidates: list[ModelCandidate],
    axis: str | None = None,
    constraints: RoutingConstraints = NO_CONSTRAINTS,
) -> tuple[ModelCandidate, list[ModelCandidate]]:
    """Return ``(selected, ordered_fallbacks)`` under the unified rule.

    Args:
        candidates:  Tier candidates, in registration order.
        axis:        Capability axis from the routing layer's task type, so
                     quality is read per-capability where a profile exists.
        constraints: Hard filters. Applied first; if nothing survives, the
                     constraint is honoured over availability and the caller
                     gets an empty result to handle explicitly.

    Raises:
        ValueError: when no candidate satisfies the constraints. Silently
            ignoring an origin restriction would be worse than failing: the
            caller asked for a guarantee, and degrading to a forbidden vendor
            breaks it invisibly.
    """
    if not candidates:
        raise ValueError("no candidates to select from")

    eligible = [c for c in candidates if constraints.permits(c)]
    if not eligible:
        excluded = ",".join(sorted(constraints.exclude_origins))
        raise ValueError(
            f"no candidate satisfies the origin constraint (excluded: {excluded}). "
            f"Tier offers: {[c.model for c in candidates]}"
        )

    tier = eligible[0].tier
    ranked = sorted(eligible, key=lambda c: -expected_value(c, axis, tier))
    return ranked[0], ranked[1:]


#: Success bar for quality-constrained selection: pick the cheapest candidate
#: whose P(success | this request) clears it.
#:
#: This exists because pricing quality does not work here. ``expected_value``
#: needs a V, and a V that is right for a throwaway summary is wrong for a
#: contract clause. Making V a per-request input would mean asking the caller
#: what a correct answer is worth to them — the same decision as asking them to
#: pick a model, which is the decision this product exists to remove. So the
#: exchange rate is replaced by a service level: one constant Theriad sets and
#: defends, with no caller input.
#:
#: 0.90 chosen from a sweep through the real harness (``run report``), not from
#: the standalone simulation — the sim's flat 500-token cost proxy understated
#: the cost of escalating by ~5x and would have selected 0.95.
#:
#:   tau    V=$0.001   V=$0.01   V=$0.10   V=$1.00     acc   cost/1k
#:   off       0.016     0.328     7.841    93.804   0.869    $0.000
#:   0.50      0.032     0.336     7.764    92.868   0.870    $0.018
#:   0.80      0.169     0.410     7.219    86.130   0.877    $0.161
#:   0.90      0.261     0.379     5.948    72.465   0.891    $0.267
#:   0.95      0.649     0.598     4.479    54.110   0.909    $0.674
#:   0.99      2.824     2.534     4.027    29.786   0.936    $2.875
#:
#: with always-cheapest at 0.024 / 0.414 / 8.708 / 102.475.
#:
#: No value dominates, and that is the point: a fixed bar cannot be right at
#: both $0.001 and $1.00 per correct answer. 0.90 wins the $0.01 and $0.10
#: columns against always-cheapest and cuts $1.00 regret by 23%, at the cost of
#: the $0.001 column (0.261 against 0.024) — escalation is a bad trade when a
#: correct answer is nearly worthless, and *every* tau gives that column up.
#: Raising this above 0.90 also loses the $0.01 column; lowering it to 0.50
#: makes the whole path barely worth its complexity.
SUCCESS_THRESHOLD = 0.90


def escalation_candidates(
    tier: str,
    registry,
    ceiling: str = "premium",
) -> list[ModelCandidate]:
    """Candidates from ``tier`` up its lane, inclusive, to ``ceiling``.

    The ladder is split into two lanes because escalating a bulk summarisation
    request into a code tier crosses task domains rather than climbing quality.
    They rejoin at premium.

    Frontier is deliberately not a default rung — ``tiers.py`` records that it
    is entered deliberately rather than as a routine escalation target, so it is
    reachable only by passing ``ceiling="frontier"`` explicitly.
    """
    lanes = (
        ["cheap_bulk", "strong_reasoning", "premium", "frontier"],
        ["cheap_code", "code_strong", "premium", "frontier"],
    )
    for lane in lanes:
        if tier in lane:
            rungs = lane[lane.index(tier):]
            if ceiling in rungs:
                rungs = rungs[: rungs.index(ceiling) + 1]
            break
    else:
        rungs = [tier]

    seen: set[str] = set()
    out: list[ModelCandidate] = []
    for rung in rungs:
        for candidate in registry.candidates_for_tier(rung) or []:
            if candidate.model not in seen:
                seen.add(candidate.model)
                out.append(candidate)
    return out


def select_by_success(
    candidates: list[ModelCandidate],
    difficulty: float,
    abilities: dict[str, float],
    threshold: float | None = None,
) -> ModelCandidate | None:
    """Cheapest candidate whose P(success) at ``difficulty`` clears ``threshold``.

    Returns None when no ability is known for any candidate — the caller must
    then fall back to ``expected_value`` rather than guess. Routing on a default
    ability would silently invent a success probability, and the whole point of
    this path is that the probability is measured.

    When abilities *are* known but none clears the bar, the most capable
    candidate is returned: the request is harder than anything available, and
    the best shot is the honest response.
    """
    from theriad.routing.difficulty import success_probability

    # Read the module constant at call time, not as a default argument: a
    # default binds at definition, so ``SUCCESS_THRESHOLD`` could not be tuned
    # by tests or a sweep without reimporting the module. That silently turned a
    # threshold sweep into five identical runs.
    bar = SUCCESS_THRESHOLD if threshold is None else threshold

    known = [c for c in candidates if c.model in abilities]
    if not known:
        return None
    for candidate in sorted(known, key=blended_cost):
        if success_probability(abilities[candidate.model], difficulty) >= bar:
            return candidate
    return max(known, key=lambda c: abilities[c.model])


def expected_value(
    candidate: ModelCandidate, axis: str | None = None, tier: str | None = None
) -> float:
    """quality x V(tier) - blended_cost. Higher is better.

    An unpriced candidate scores -inf via :func:`blended_cost`, so it is only
    ever selected when nothing else survives the constraints.
    """
    quality = candidate.quality_for(axis) or 0.0
    return quality * value_of_quality(tier or candidate.tier) - blended_cost(candidate)
