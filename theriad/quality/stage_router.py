"""
Stage router for quality-loop refinement passes.

Maps each loop stage + task type to the appropriate Theriad routing tier.
Routing follows the existing tier abstraction (never model slugs).

Defaults
--------
draft    → whatever the normal routing decision was (passed in)
critique → cheap_reasoning (escalates to strong_reasoning on high risk/ambiguity)
repair   → cheap_bulk (prose), code_agent (code), strong_reasoning (reasoning tasks)
validate → no_model (deterministic tools first); falls back to cheap_reasoning
final    → cheap_bulk (synthesis is usually simpler than repair)
"""
from __future__ import annotations

from dataclasses import dataclass

from .loop_schemas import LoopStage


# Task-type groups
_CODE_TASKS     = {"code", "json", "html"}
_REASONING_TASKS = {"research", "analysis", "planning"}


@dataclass(frozen=True)
class StageRouteDecision:
    stage: LoopStage
    tier: str
    use_validation_tool: bool   # True when a deterministic tool should run first
    reason: str


def route_stage(
    stage: LoopStage,
    *,
    task_type: str = "general",
    risk_level: str = "low",           # low | medium | high
    prior_failures: int = 0,
    draft_tier: str = "cheap_bulk",    # tier used for the initial draft
) -> StageRouteDecision:
    """
    Return the tier to use for a given loop stage.

    Parameters are all keyword-only after `stage` to prevent positional mistakes.
    """
    if stage == LoopStage.DRAFT:
        return StageRouteDecision(
            stage=stage,
            tier=draft_tier,
            use_validation_tool=False,
            reason="draft uses the same tier as the original routing decision",
        )

    if stage == LoopStage.CRITIQUE:
        if risk_level in ("high",) or prior_failures >= 2:
            tier = "strong_reasoning"
            reason = "escalated critique: high risk or repeated failures"
        else:
            tier = "cheap_reasoning"
            reason = "cheap_reasoning is sufficient for initial critique"
        return StageRouteDecision(
            stage=stage,
            tier=tier,
            use_validation_tool=False,
            reason=reason,
        )

    if stage == LoopStage.REPAIR:
        if task_type in _CODE_TASKS:
            tier = "code_agent"
            reason = "code repair routed to code_agent tier"
        elif task_type in _REASONING_TASKS or risk_level == "high":
            tier = "strong_reasoning"
            reason = "reasoning-heavy repair escalated to strong_reasoning"
        else:
            tier = "cheap_bulk"
            reason = "prose/general repair uses cheap_bulk"
        return StageRouteDecision(
            stage=stage,
            tier=tier,
            use_validation_tool=False,
            reason=reason,
        )

    if stage == LoopStage.VALIDATE:
        # Prefer deterministic tools; fall back to cheap_reasoning only if needed.
        return StageRouteDecision(
            stage=stage,
            tier="no_model",
            use_validation_tool=True,
            reason="validation prefers deterministic tools over model calls",
        )

    if stage == LoopStage.FINAL:
        if task_type in _REASONING_TASKS:
            tier = "cheap_reasoning"
            reason = "reasoning synthesis uses cheap_reasoning for final pass"
        else:
            tier = "cheap_bulk"
            reason = "final synthesis uses cheap_bulk"
        return StageRouteDecision(
            stage=stage,
            tier=tier,
            use_validation_tool=False,
            reason=reason,
        )

    # Fallback
    return StageRouteDecision(
        stage=stage,
        tier="cheap_bulk",
        use_validation_tool=False,
        reason=f"unknown stage '{stage}' — defaulting to cheap_bulk",
    )
