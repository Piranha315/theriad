"""
Soft conversation continuity signal for model routing.

Design decisions:
- Continuity is a SOFT bonus only.  It never overrides:
    * The tier selected by the routing layers (tier floor).
    * Context-window fit (a model that can't hold the prompt is excluded first).
    * Explicit user model choice.
    * A cost or latency difference large enough to cross a tier boundary — the
      bonus is intentionally small relative to cross-tier cost deltas.
- Bonus weights are policy-differentiated.  aggressive_savings keeps the weight
  near zero so cost remains the dominant signal; balanced allows a moderate push
  toward continuity; quality_first is below balanced because quality outranks
  coherence.
- Task family matching (same_task_family) drives the bonus.  When the task
  family is different, the bonus is zero regardless of policy — context
  switching is a natural model-switch signal.
- The bonus for the same task sub-type is the full weight; same family but
  different sub-type gets half weight.
- score_continuity() is a pure function with no side effects so it can be
  tested and replaced independently.

Core product principle (verbatim):
    "Theriad should minimize unnecessary cost.  Do not add optimizations that
    make the app feel more coherent or lower-latency by spending more by
    default.  Any continuity or context-window behavior should be a soft
    routing signal, not a hard constraint."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Bonus weights by policy ───────────────────────────────────────────────────
#
# These must be SMALLER than the typical score difference between policy
# selections (e.g., cost difference between cheap_code models) so that the
# bonus cannot flip a policy decision across a tier boundary.
#
# Scale: quality_fit / cost scores are typically in [0, 1]; continuity bonus
# is a fractional addend in that same space.

_BONUS_WEIGHT_BY_POLICY: dict[str, float] = {
    "aggressive_savings": 0.04,  # cost dominates; continuity is almost noise
    "balanced":           0.15,  # moderate push for coherence
    "quality_first":      0.08,  # quality dominates; continuity is secondary
}
_DEFAULT_BONUS_WEIGHT: float = 0.10


# ── Types ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContinuitySignal:
    """
    Inputs derived from the conversation's sticky-route entry.

    Populated by StickyRouteCache.get_continuity_signal(); callers should
    treat all fields as optional — a cold conversation has no signal.

    Fields:
        previous_model:               Model slug used in the last turn.
        previous_tier:                Tier label from the last turn.
        previous_task_type:           Task-type hint from the last turn (free text).
        current_task_type:            Task-type hint for the current turn.
        conversation_summary_available: True if a stored summary is available
                                       (reserved for future RAG use; not used yet).
        same_task_family:             Caller-derived: True when the routing tier
                                       difference is < _MAJOR_TIER_DELTA, i.e. the
                                       conversation has not shifted work category.
    """

    previous_model: Optional[str]
    previous_tier: Optional[str]
    previous_task_type: Optional[str] = None
    current_task_type: Optional[str] = None
    conversation_summary_available: bool = False
    same_task_family: bool = False


@dataclass(frozen=True)
class ContinuityScore:
    """
    Result of scoring one candidate against the continuity signal.

    Fields:
        bonus:   Score to add to this candidate's base score.  Zero when
                 continuity does not apply.
        applied: True when a non-zero bonus was added.
        weight:  Policy-derived maximum bonus weight (for diagnostics).
        reason:  Human-readable explanation for the decision log.
    """

    bonus: float
    applied: bool
    weight: float
    reason: str


# ── Scorer ────────────────────────────────────────────────────────────────────


def score_continuity(
    signal: ContinuitySignal,
    candidate_model: str,
    policy: str,
) -> ContinuityScore:
    """
    Compute a soft continuity bonus for a single candidate model.

    Returns a ContinuityScore with bonus=0 when continuity does not apply.

    Constraints enforced here:
    - No previous model → bonus=0.
    - Candidate is not the previous model → bonus=0.
    - Different task family → bonus=0 (task switch is a natural model-switch).
    - Same family, same sub-type → full bonus.
    - Same family, different sub-type → half bonus.

    Context-window and tier-floor constraints are enforced upstream
    (in PolicyResolver) before score_continuity is called.
    """
    weight = _BONUS_WEIGHT_BY_POLICY.get(policy, _DEFAULT_BONUS_WEIGHT)

    if not signal.previous_model:
        return ContinuityScore(
            bonus=0.0,
            applied=False,
            weight=weight,
            reason="no_prior_model",
        )

    if candidate_model != signal.previous_model:
        return ContinuityScore(
            bonus=0.0,
            applied=False,
            weight=weight,
            reason="different_model",
        )

    # Candidate IS the previous model — check task family
    if not signal.same_task_family:
        return ContinuityScore(
            bonus=0.0,
            applied=False,
            weight=weight,
            reason="different_task_family_no_bonus",
        )

    # Same task family → apply bonus
    if (
        signal.previous_task_type
        and signal.current_task_type
        and signal.previous_task_type == signal.current_task_type
    ):
        return ContinuityScore(
            bonus=weight,
            applied=True,
            weight=weight,
            reason=f"same_task_type:{signal.current_task_type}",
        )

    # Same family but sub-type unknown or changed → half bonus
    return ContinuityScore(
        bonus=weight * 0.5,
        applied=True,
        weight=weight,
        reason="same_task_family_different_type",
    )
