"""
Policy resolver: maps (tier, policy) → concrete ModelCandidate.

Design decisions:
- Three policies are implemented via a unified scoring function so that the
  continuity bonus can be layered on top without duplicating selection logic.
- aggressive_savings: score by cost (lower = higher score); free models get a
  large bonus when confidence ≥ 0.8.
- balanced: primary candidate gets a score bonus; role=='primary' is the
  tiebreaker.
- quality_first: score = quality on the request's capability axis, falling back
  to the aggregate quality_prior when the axis is unknown or unmeasured
  (see theriad.models.capabilities).
- Epsilon-greedy exploration is opt-in and off by default; when enabled it can
  override the winner *within* a tier so the router keeps collecting evidence
  about candidates its priors currently disfavour (see theriad.models.exploration).
- Context-window filtering: candidates whose context_window < required_tokens
  are excluded before scoring.  If none survive, all candidates are retained
  (provider will surface the real error).
- Continuity is a soft bonus added to the score of the previously-used model.
  It can never flip a decision across a tier boundary — bonus weights are
  kept smaller than typical inter-policy score deltas.
- Resolution never hard-fails: if a tier has no candidates, it falls through
  to cheap_bulk as an emergency default and logs a warning.

Policy behavior summary:
  aggressive_savings  — cheapest cost, free models preferred, quality secondary
  balanced            — primary candidate per tier; reliable, predictable
  quality_first       — highest quality_prior regardless of cost
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from theriad.core.logging.logger import get_logger

from .capabilities import axis_for_task_type
from .exploration import NO_EXPLORATION, ExplorationConfig
from .registry import DEFAULT_REGISTRY, ModelRegistry
from .tiers import TIER_CHEAP_BULK, ModelCandidate

if TYPE_CHECKING:
    from theriad.routing.context import ContextEstimate
    from theriad.routing.continuity import ContinuitySignal

logger = get_logger("models.policy")

VALID_POLICIES = frozenset({
    "aggressive_savings",
    "balanced",
    "quality_first",
    "free_moe_experimental",
})

# Numeric cost/quality tradeoff value per policy name.
# Scale: 0 = pure quality, 10 = pure cost savings.
# Used in run metadata and benchmark results so dashboards can display intent.
POLICY_TRADEOFF: dict[str, int] = {
    "aggressive_savings": 9,
    "balanced":           7,
    "quality_first":      2,
    "free_moe_experimental": 10,
}


@dataclass
class PolicyResolution:
    """Result of resolving a tier + policy to a concrete model."""
    tier: str
    selected: ModelCandidate
    fallback_chain: list[ModelCandidate]
    policy: str

    # ── Route diagnostics ─────────────────────────────────────────────────────
    # Populated when context_estimate / continuity_signal are provided to resolve().
    # Defaults to None/False/empty so existing call sites need no changes.
    selected_context_bucket: Optional[str] = field(default=None)
    selected_model_context_limit: Optional[int] = field(default=None)
    context_fit_reason: Optional[str] = field(default=None)
    compression_or_chunking_recommended: bool = field(default=False)
    continuity_bonus_applied: bool = field(default=False)
    continuity_bonus_weight: float = field(default=0.0)
    reason_for_stay_or_switch: str = field(default="")
    speed_preference: int = 50
    quality_preference: int = 50
    estimated_price_multiplier: float = 1.0

    # ── Capability routing / exploration diagnostics ──────────────────────────
    capability_axis: Optional[str] = field(default=None)
    """Axis used to score quality, or None when scored on the aggregate prior."""

    exploration_applied: bool = field(default=False)
    """True when epsilon-greedy overrode the top-scored candidate.

    Outcome logging MUST persist this. Exploit rows are selected *by the current
    policy*, so treating them as a random sample of model performance will
    reliably confirm whatever the policy already believed.
    """

    exploit_model: Optional[str] = field(default=None)
    """The model that would have been chosen had this not been an explore step."""

    # Router V2 diagnostics. They remain empty when V2 is off or its artifact
    # cannot be loaded, which is the fail-closed compatibility path.
    router_version: Optional[str] = field(default=None)
    lane: Optional[str] = field(default=None)
    knee_threshold: Optional[float] = field(default=None)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    ood_score: Optional[float] = field(default=None)
    selected_propensity: Optional[float] = field(default=None)
    shadow: bool = field(default=False)
    shadow_challenger: Optional[str] = field(default=None)
    router_v2_decision: Any = field(default=None, repr=False)

    @property
    def provider(self) -> str:
        return self.selected.provider

    @property
    def model(self) -> str:
        return self.selected.model

    @property
    def cost_quality_tradeoff(self) -> int:
        """Numeric tradeoff value for this policy (0=quality, 10=savings)."""
        return POLICY_TRADEOFF.get(self.policy, POLICY_TRADEOFF["balanced"])

    def fallback_tuples(self) -> list[tuple[str, str]]:
        """Return [(provider, model), …] for the TheriadPipeline fallback chain."""
        return [(c.provider, c.model) for c in self.fallback_chain]


class PolicyResolver:
    """
    Resolves a routing tier + policy name to a concrete (provider, model) pair.

    Usage:
        resolver = PolicyResolver()
        resolution = resolver.resolve("cheap_code", "balanced")
        print(resolution.provider, resolution.model)
    """

    def __init__(self, registry: ModelRegistry | None = None):
        self._registry = registry or DEFAULT_REGISTRY

    def resolve(
        self,
        tier: str,
        policy: str = "balanced",
        confidence: float = 1.0,
        context_estimate: Optional["ContextEstimate"] = None,
        continuity_signal: Optional["ContinuitySignal"] = None,
        message: str = "",
        speed_preference: int | None = None,
        quality_preference: int | None = None,
        task_type: str | None = None,
        exploration: ExplorationConfig | None = None,
        constraints: "RoutingConstraints | None" = None,
        escalation: bool = True,
        escalation_ceiling: str = "premium",
    ) -> PolicyResolution:
        """
        Select a ModelCandidate from the tier using the given policy.

        Args:
            tier:               Tier label emitted by a routing layer.
            policy:             One of "aggressive_savings", "balanced",
                                "quality_first".
            confidence:         Routing confidence [0, 1]; used by
                                aggressive_savings to decide whether to risk a
                                free-tier model.
            context_estimate:   Optional output of ContextRequirementEstimator.
                                When provided, candidates whose context_window
                                is smaller than total_context_required are
                                excluded before policy scoring.
            continuity_signal:  Optional signal from the previous conversation
                                turn.  When provided, the previously-used model
                                receives a small policy-weighted score bonus.
                                The bonus is NEVER large enough to override tier
                                floor, context fit, or a clear cost difference.
            message:            Raw user message used only by the opt-in
                                free_moe_experimental classifier. Other
                                policies ignore it.
            task_type:          Optional task label from the routing layer
                                (RoutingDecision.signals["task_type"]). Maps to a
                                capability axis so quality is scored on the kind
                                of work being routed rather than an aggregate.
                                None reproduces pre-capability behaviour exactly.
            exploration:        Optional epsilon-greedy config. Off by default;
                                when enabled it may override the top-scored
                                candidate within this tier. Never crosses tiers.
            escalation:         Allow selection to climb above ``tier`` when the
                                request looks too hard for it. On by default.
                                Requires both a difficulty model and measured
                                abilities; without either it is inert and this
                                reduces to within-tier expected value, which is
                                the behaviour that shipped before. Pass False to
                                pin selection to ``tier`` regardless.
            escalation_ceiling: Highest tier escalation may reach. Defaults to
                                premium; frontier must be asked for explicitly
                                (see tiers.py).

        Returns:
            PolicyResolution with the selected candidate, ordered fallback
            chain, and populated route diagnostics.
        """
        from .preferences import RoutingPreferences

        preferences = RoutingPreferences.resolve(
            speed=speed_preference,
            quality=quality_preference,
            legacy_policy=policy,
        )

        if policy not in VALID_POLICIES:
            logger.warning("unknown policy, defaulting to balanced", extra={"policy": policy})
            policy = "balanced"

        if policy == "free_moe_experimental":
            from theriad.models.free_moe import FreeMoeResolver
            return FreeMoeResolver(self._registry).resolve(
                message=message,
                confidence=confidence,
                context_estimate=context_estimate,
                continuity_signal=continuity_signal,
            )

        candidates = self._registry.candidates_for_tier(tier)
        if not candidates:
            logger.warning(
                "tier has no candidates, falling back to cheap_bulk",
                extra={"tier": tier},
            )
            tier = TIER_CHEAP_BULK
            candidates = self._registry.candidates_for_tier(tier)

        if not candidates:
            raise RuntimeError(f"PolicyResolver: no candidates found for tier '{tier}'")

        # ── Context-window filtering ──────────────────────────────────────────
        context_fit_reason: Optional[str] = None
        compression_recommended = False

        if context_estimate is not None:
            from theriad.routing.context import filter_candidates_by_context
            filtered = filter_candidates_by_context(
                candidates, context_estimate.total_context_required
            )
            if filtered is not candidates:
                excluded = len(candidates) - len(filtered)
                logger.debug(
                    "context filter excluded candidates",
                    extra={
                        "tier": tier,
                        "excluded": excluded,
                        "required_tokens": context_estimate.total_context_required,
                    },
                )
            candidates = filtered
            context_fit_reason = context_estimate.context_fit_reason
            compression_recommended = context_estimate.compression_or_chunking_recommended

        # ── Unified selection (D-054) ─────────────────────────────────────────
        # One rule for every request: maximise quality x V(tier) - cost, subject
        # to hard constraints. `policy`, `speed_preference` and
        # `quality_preference` are accepted for wire compatibility and no longer
        # affect the outcome — see theriad/models/unified.py for why the dial
        # was removed rather than retuned.
        from .unified import (
            NO_CONSTRAINTS,
            escalation_candidates,
            expected_value,
            select_by_success,
        )

        axis = axis_for_task_type(task_type)
        active_constraints = constraints or NO_CONSTRAINTS

        # ── Quality-constrained escalation (D-055) ────────────────────────────
        # Widen the candidate pool up the lane, then pick the cheapest model
        # whose measured P(success) on *this* prompt clears the bar. This is the
        # only path that differentiates per request: scoring on quality x V
        # picks the same model for every request, because the argmax does not
        # depend on the request.
        #
        # Both inputs must be present. A missing difficulty model or missing
        # abilities means no measured success probability exists, and inventing
        # one would be worse than not escalating — so the code falls through to
        # the within-tier expected-value path below, unchanged.
        escalated: ModelCandidate | None = None
        if escalation:
            from theriad.routing.difficulty import load_difficulty_model

            from .calibration import load_calibration

            abilities = load_calibration().ability
            difficulty_model = load_difficulty_model()
            if abilities and not difficulty_model.is_empty and message:
                pool = [
                    c
                    for c in escalation_candidates(
                        tier, self._registry, escalation_ceiling
                    )
                    if active_constraints.permits(c)
                ]
                if pool:
                    escalated = select_by_success(
                        pool, difficulty_model.predict(message), abilities
                    )
                    if escalated is not None and escalated.tier != tier:
                        logger.info(
                            "escalated above routed tier",
                            extra={
                                "from_tier": tier,
                                "to_tier": escalated.tier,
                                "model": escalated.model,
                            },
                        )

        permitted = [c for c in candidates if active_constraints.permits(c)]
        if not permitted:
            raise RuntimeError(
                f"tier '{tier}' has no candidate satisfying the origin constraint "
                f"(excluded: {sorted(active_constraints.exclude_origins)}). "
                f"Offers: {[c.model for c in candidates]}. Honouring the "
                f"constraint over availability — degrading to an excluded vendor "
                f"would break the guarantee invisibly."
            )
        candidates = permitted
        scores = [expected_value(c, axis, tier) for c in candidates]

        # ── Apply soft continuity bonus ───────────────────────────────────────
        continuity_bonus_applied = False
        continuity_bonus_weight = 0.0
        reason_for_stay_or_switch = "fresh_routing"

        if continuity_signal is not None and continuity_signal.previous_model:
            from theriad.routing.continuity import score_continuity
            for i, candidate in enumerate(candidates):
                cs = score_continuity(continuity_signal, candidate.model, policy)
                if cs.applied:
                    scores[i] += cs.bonus
                    continuity_bonus_applied = True
                    continuity_bonus_weight = cs.weight
                    reason_for_stay_or_switch = f"continuity_bonus:{cs.reason}"
                elif not continuity_bonus_applied and i == 0:
                    # Record why bonus was NOT applied (first candidate's reason)
                    reason_for_stay_or_switch = f"no_continuity:{cs.reason}"

        # ── Select highest-scored candidate ───────────────────────────────────
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        exploit_idx = best_idx

        # ── Epsilon-greedy exploration (opt-in, within-tier only) ─────────────
        explore_cfg = exploration or NO_EXPLORATION
        exploration_applied = False
        if explore_cfg.enabled and len(candidates) > 1 and explore_cfg.should_explore():
            best_idx = explore_cfg.choose_index(len(candidates), exploit_idx)
            exploration_applied = best_idx != exploit_idx
            if exploration_applied:
                logger.info(
                    "exploration override",
                    extra={
                        "tier": tier,
                        "epsilon": explore_cfg.epsilon,
                        "exploit_model": candidates[exploit_idx].model,
                        "explore_model": candidates[best_idx].model,
                    },
                )

        selected = candidates[best_idx]
        fallback = [
            candidates[i]
            for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            if candidates[i] is not selected
        ]

        # ── Escalation override ───────────────────────────────────────────────
        # Applied last, on purpose. The within-tier path above runs untouched,
        # so continuity and exploration keep operating over a single tier —
        # exploration in particular is documented never to cross tiers, and a
        # widened pool would break that. The tier-local winner is demoted to the
        # head of the fallback chain rather than discarded: if the escalated
        # model fails, the next attempt should be what the tier would have
        # chosen anyway.
        resolved_tier = tier
        if escalated is not None and escalated is not selected:
            fallback = [selected] + [c for c in fallback if c.model != escalated.model]
            selected = escalated
            # Report the tier actually used, so per-tier output caps and
            # timeouts (TIER_MAX_TOKENS, TIER_TIMEOUT_SECONDS) match the model
            # that will run. Keeping the routed tier here would cap a premium
            # model at cheap_bulk's 1200 tokens and truncate it.
            resolved_tier = escalated.tier

        # ── Artifact-backed Router V2 rollout ────────────────────────────────
        # Run after the legacy resolver has produced a complete baseline. In
        # shadow mode V2 only records its counterfactual; in canary/on it may
        # replace the selected model. Any artifact/configuration error returns
        # None and leaves the baseline byte-for-byte intact.
        v2_decision = None
        if message:
            from theriad.routing.v2 import resolve_from_environment

            v2_pool = [
                candidate
                for candidate in escalation_candidates(tier, self._registry, escalation_ceiling)
                if active_constraints.permits(candidate)
            ]
            if context_estimate is not None:
                required = context_estimate.total_context_required
                v2_pool = [
                    candidate
                    for candidate in v2_pool
                    if candidate.context_window is None or candidate.context_window >= required
                ]
            from theriad.runtime.circuit_breaker import unavailable_model_slugs

            v2_decision = resolve_from_environment(
                message,
                v2_pool,
                tier=tier,
                baseline_model=selected.model,
                unavailable_models=unavailable_model_slugs(v2_pool),
            )
            if v2_decision is not None and v2_decision.applied:
                v2_selected = next(
                    (candidate for candidate in v2_pool if candidate.model == v2_decision.selected_model),
                    None,
                )
                if v2_selected is not None and v2_selected is not selected:
                    previous = selected
                    selected = v2_selected
                    resolved_tier = selected.tier
                    fallback = [previous] + [
                        candidate
                        for candidate in fallback + v2_pool
                        if candidate.model not in {selected.model, previous.model}
                    ]
                    # A model may appear in multiple rungs; provider fallbacks
                    # must not retry the same concrete slug.
                    seen_fallbacks: set[str] = set()
                    unique_fallbacks: list[ModelCandidate] = []
                    for candidate in fallback:
                        if candidate.model not in seen_fallbacks:
                            seen_fallbacks.add(candidate.model)
                            unique_fallbacks.append(candidate)
                    fallback = unique_fallbacks

        # If continuity bonus was applied but didn't change the winner, update reason
        if (
            continuity_bonus_applied
            and continuity_signal is not None
            and selected.model != continuity_signal.previous_model
        ):
            reason_for_stay_or_switch = (
                f"switched_model:{continuity_signal.previous_model}→{selected.model}"
                f" (bonus insufficient to retain previous)"
            )

        logger.debug(
            "policy resolved",
            extra={
                "tier": tier,
                "policy": policy,
                "model": selected.model,
                "fallbacks": len(fallback),
                "continuity_bonus_applied": continuity_bonus_applied,
                "context_bucket": context_estimate.bucket.value if context_estimate else None,
                "capability_axis": axis,
                "exploration_applied": exploration_applied,
            },
        )

        return PolicyResolution(
            tier=resolved_tier,
            selected=selected,
            fallback_chain=fallback,
            policy=policy,
            selected_context_bucket=(
                context_estimate.bucket.value if context_estimate else None
            ),
            selected_model_context_limit=selected.context_window,
            context_fit_reason=context_fit_reason,
            compression_or_chunking_recommended=compression_recommended,
            continuity_bonus_applied=continuity_bonus_applied,
            continuity_bonus_weight=continuity_bonus_weight,
            reason_for_stay_or_switch=reason_for_stay_or_switch,
            speed_preference=preferences.speed,
            quality_preference=preferences.quality,
            estimated_price_multiplier=preferences.price_multiplier,
            capability_axis=axis,
            exploration_applied=exploration_applied,
            exploit_model=candidates[exploit_idx].model,
            router_version=v2_decision.router_version if v2_decision else None,
            lane=v2_decision.lane if v2_decision else None,
            knee_threshold=v2_decision.knee_threshold if v2_decision else None,
            candidate_scores=(
                [prediction.to_dict() for prediction in v2_decision.candidate_scores]
                if v2_decision
                else []
            ),
            ood_score=v2_decision.ood_score if v2_decision else None,
            selected_propensity=v2_decision.selected_propensity if v2_decision else None,
            shadow=v2_decision.shadow if v2_decision else False,
            shadow_challenger=v2_decision.shadow_challenger if v2_decision else None,
            router_v2_decision=v2_decision,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_candidates(
        self,
        candidates: list[ModelCandidate],
        policy: str,
        confidence: float,
        speed_preference: int | None = None,
        quality_preference: int | None = None,
        axis: str | None = None,
    ) -> list[float]:
        """
        Return a parallel list of scores for each candidate under the policy.

        Higher score = preferred.  Scores are on an arbitrary scale; only
        relative ordering matters.  Continuity bonus is added by the caller.

        ``axis`` selects which capability score quality-sensitive policies read;
        None means score on the aggregate ``quality_prior``.
        """
        if speed_preference is not None or quality_preference is not None:
            return self._scores_continuous_preferences(
                candidates,
                speed_preference=50 if speed_preference is None else speed_preference,
                quality_preference=50 if quality_preference is None else quality_preference,
                axis=axis,
            )
        if policy == "aggressive_savings":
            return self._scores_aggressive_savings(candidates, confidence)
        if policy == "quality_first":
            return self._scores_quality_first(candidates, axis)
        return self._scores_balanced(candidates)

    @staticmethod
    def _scores_continuous_preferences(
        candidates: list[ModelCandidate],
        *,
        speed_preference: int,
        quality_preference: int,
        axis: str | None = None,
    ) -> list[float]:
        """Score actual candidates on normalized cost, quality, and latency.

        Quality is the dominant model-selection control. Speed contributes only
        when empirical latency priors exist; otherwise endpoint routing handles
        it, avoiding invented model-speed claims.
        """
        from .preferences import clamp_preference

        quality_weight = clamp_preference(quality_preference) / 100
        speed_weight = clamp_preference(speed_preference) / 100
        blended_costs = [
            (
                0.75 * c.input_cost_per_million + 0.25 * c.output_cost_per_million
                if c.input_cost_per_million is not None
                and c.output_cost_per_million is not None
                else None
            )
            for c in candidates
        ]
        known_costs = [cost for cost in blended_costs if cost is not None]
        max_cost = max(known_costs, default=1.0) or 1.0
        known_latencies = [
            c.latency_p50_ms for c in candidates if c.latency_p50_ms is not None
        ]
        max_latency = max(known_latencies, default=1.0) or 1.0

        scores: list[float] = []
        for candidate, cost in zip(candidates, blended_costs, strict=True):
            cost_score = 0.0 if cost is None else 1.0 - min(cost / max_cost, 1.0)
            quality_score = candidate.quality_for(axis) or 0.0
            latency_score = (
                1.0 - min(candidate.latency_p50_ms / max_latency, 1.0)
                if candidate.latency_p50_ms is not None
                else 0.5
            )
            role_bonus = 0.08 if candidate.role == "primary" else 0.0
            scores.append(
                (1.0 - quality_weight) * cost_score
                + quality_weight * quality_score
                + 0.15 * speed_weight * latency_score
                + role_bonus
            )
        return scores

    @staticmethod
    def _scores_aggressive_savings(
        candidates: list[ModelCandidate], confidence: float
    ) -> list[float]:
        """Lower cost → higher score.  Free models with high confidence win."""
        scores = []
        for c in candidates:
            cost = c.input_cost_per_million
            if cost is None:
                score = -500.0   # unknown cost: treat as expensive, avoid
            elif cost == 0.0 and confidence >= 0.80:
                score = 1000.0   # free model at high confidence: top priority
            else:
                score = -(cost if cost >= 0 else 999.0)
            scores.append(score)
        return scores

    @staticmethod
    def _scores_balanced(candidates: list[ModelCandidate]) -> list[float]:
        """Primary role gets 1.0; all others get 0.5."""
        return [1.0 if c.role == "primary" else 0.5 for c in candidates]

    @staticmethod
    def _scores_quality_first(
        candidates: list[ModelCandidate], axis: str | None = None
    ) -> list[float]:
        """Score = quality on ``axis``; unknown quality treated as 0.0.

        With axis=None this is the historical behaviour (aggregate prior). With
        an axis, candidates lacking a measurement on it fall back to their
        aggregate rather than scoring zero — see ModelCandidate.quality_for.
        """
        return [c.quality_for(axis) or 0.0 for c in candidates]
