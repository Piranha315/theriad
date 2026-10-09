"""
Capability axes: per-benchmark quality signals for within-tier model selection.

Why this exists
---------------
``ModelCandidate.quality_prior`` is a single scalar. A scalar can only express
"model A is better than model B", which is the wrong question when candidates
sit within a point or two of each other on an aggregate index. Aggregate scores
average over task families that models are *not* uniformly good at: a model can
lead on agentic coding and trail on knowledge recall while the aggregate hides
both facts.

An axis is a task family with a benchmark family behind it. ``quality_profile``
on a ModelCandidate maps axis → score in [0, 1], so the resolver can ask "who is
best at *this kind of work*" instead of "who is best".

Relationship to tiers (D-016 / D-017)
-------------------------------------
This does NOT let routing layers name models — layers still emit tier labels
only, and the resolver still owns tier → model. What changes is that the
resolver gains a second input (the task type the layer already computed and
already ships in ``RoutingDecision.signals``), so a tier becomes closer to a
*cost band* within which capability decides, rather than a rung on a single
quality ladder.

Provenance rules
----------------
Populating a profile with a number nobody measured is worse than leaving it
empty, because ``quality_for()`` silently falls back to the scalar prior when an
axis is missing — a documented, conservative default — whereas an invented axis
score routes traffic on fiction. Every profile MUST set
``quality_profile_source``. Prefer leaving an axis absent over guessing it.
"""
from __future__ import annotations

# ── Axis constants ────────────────────────────────────────────────────────────

AXIS_CODE = "code"
"""Software engineering: multi-file edits, debugging, repo-level reasoning.
Benchmark families: SWE-bench, LiveCodeBench, HumanEval+, AA Coding Index."""

AXIS_REASONING = "reasoning"
"""Multi-step analytical reasoning, math, planning.
Benchmark families: GPQA, AIME, MATH-500, MMLU-Pro, AA Intelligence Index."""

AXIS_KNOWLEDGE = "knowledge"
"""Factual recall and open-domain question answering.
Benchmark families: HLE, SimpleQA, MMLU-Pro (knowledge subsets)."""

AXIS_BULK = "bulk"
"""High-volume structured transforms: extraction, summarization, rewriting.
Benchmark families: JSON schema validity, exact-match extraction, FLORES-200."""

VALID_AXES: frozenset[str] = frozenset({
    AXIS_CODE,
    AXIS_REASONING,
    AXIS_KNOWLEDGE,
    AXIS_BULK,
})


# ── task_type → axis ──────────────────────────────────────────────────────────
#
# Keys cover both vocabularies already in the codebase:
#   - EmbeddingRouter prototypes (theriad/layers/layer2_embedding/router.py)
#   - WorkflowCostSimulator task types (theriad/workflow/cost_simulator.py)
# Unmapped task types resolve to None, which makes the resolver fall back to the
# scalar prior — the pre-existing behaviour.

_TASK_TYPE_TO_AXIS: dict[str, str] = {
    # EmbeddingRouter prototype labels
    "code":          AXIS_CODE,
    "code_hard":     AXIS_CODE,
    "qa":            AXIS_KNOWLEDGE,
    "summarize":     AXIS_BULK,
    "translate":     AXIS_BULK,
    "analysis":      AXIS_REASONING,
    # WorkflowCostSimulator task types
    "coding":        AXIS_CODE,
    "debugging":     AXIS_CODE,
    "architecture":  AXIS_REASONING,
    "planning":      AXIS_REASONING,
    "reasoning":     AXIS_REASONING,
    "summarization": AXIS_BULK,
    "extraction":    AXIS_BULK,
    "translation":   AXIS_BULK,
    "rewriting":     AXIS_BULK,
    "documentation": AXIS_BULK,
}


def axis_for_task_type(task_type: str | None) -> str | None:
    """Return the capability axis for a task type, or None if unmapped.

    None is a meaningful result, not an error: it tells the resolver to score on
    the scalar ``quality_prior`` exactly as it did before capability profiles
    existed.
    """
    if not task_type:
        return None
    return _TASK_TYPE_TO_AXIS.get(task_type.strip().lower())


# ── Normalizing published benchmark scores ────────────────────────────────────
#
# This is what blocked profile population until now. `quality_for()` falls back
# to `quality_prior` when an axis is unmeasured, so profile scores and scalar
# priors get compared inside a single ranking — which means they must share a
# scale. The shipped priors were hand-authored on a 0.68–0.92 band and, where
# published data exists, they disagreed with it: the retired Kimi K2.7 Code
# carried a 0.88 prior against an Artificial Analysis Intelligence Index of 42
# (second-lowest measured), while GLM-5.2 carries 0.92 at index 51.
#
# The fix is one documented transform from the published index onto the band the
# registry already uses, so measured and unmeasured candidates remain comparable
# and every future addition lands on the same scale.

AA_INDEX_FLOOR = 42.0
"""Lowest Artificial Analysis Intelligence Index among registry candidates."""

AA_INDEX_CEILING = 61.0
"""Highest measured index (Claude Opus 5 at max reasoning effort)."""

QUALITY_FLOOR = 0.75
QUALITY_CEILING = 0.95
"""Target band. Chosen to sit inside the registry's existing 0.68–0.92 spread so
converted scores stay comparable against candidates that have no published
index yet.

Widening this band looks tempting — it compresses the 0–100% capability range
into 0.20 of quality — but it is not a bug and widening it would double-count.
``theriad.eval.calibrate`` derives the router's V as
``v_oracle / (scale.slope * tokens)``, so a narrow band produces a
proportionally larger V and the product is unchanged. Compression here is
compensated there by construction."""


def index_to_quality(aa_index: float) -> float:
    """Map an Artificial Analysis Intelligence Index onto the registry's scale.

    Affine and monotone, clipped to the band:

        42 -> 0.750    51 -> 0.845    56 -> 0.897    61 -> 0.950
        44 -> 0.771    54 -> 0.876    59 -> 0.929

    Deliberately linear rather than fitted. There are only two overlapping data
    points between the published index and the hand-authored priors, and they
    disagree with each other — fitting a curve to that would be fitting noise
    and would dress a guess up as a calibration.

    Use this for **general-purpose** models, where an aggregate index is a fair
    summary. Do not use it for specialists: a code model can score poorly on a
    general index while being excellent at the one axis it is routed for, which
    is the whole reason per-axis profiles exist.
    """
    span_in = AA_INDEX_CEILING - AA_INDEX_FLOOR
    span_out = QUALITY_CEILING - QUALITY_FLOOR
    scaled = QUALITY_FLOOR + (aa_index - AA_INDEX_FLOOR) * span_out / span_in
    return round(min(max(scaled, 0.0), 1.0), 3)
