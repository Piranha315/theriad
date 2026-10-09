"""
Browser-agent subtask router.

Maps each agent subtask type to an abstract tier (not a concrete model slug).
The router never makes provider calls — it only returns routing decisions.

Design:
- Tiers are abstract strings; concrete model selection happens in the pipeline.
- Screenshot/vision tiers are optional and not called by default.
- Cheap tiers are used aggressively; strong tiers only when escalation triggers.
- New agent tiers (cheap_reasoning, vision_cheap, vision_strong) are defined here
  and mapped conservatively to existing tiers when no dedicated models exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .browser_schemas import (
    SUBTASK_OBSERVE_DOM,
    SUBTASK_OBSERVE_SCREENSHOT,
    SUBTASK_EXTRACT_DATA,
    SUBTASK_PLAN_NEXT_ACTION,
    SUBTASK_VALIDATE_ACTION,
    SUBTASK_SUMMARIZE_LOG,
    SUBTASK_FINAL_REPORT,
    SUBTASK_ORCHESTRATE,
    SUBTASK_EVALUATE_GOAL,
    POLICY_AGGRESSIVE_SAVINGS,
    POLICY_BALANCED,
    POLICY_QUALITY_FIRST,
    POLICY_FAST,
)

# ── Agent-specific abstract tier names ───────────────────────────────────────
# These map to existing Theriad tiers conservatively.
# Future: give each its own registry entry.

TIER_CHEAP_REASONING = "cheap_reasoning"   # → cheap_code (reasoning-capable cheap model)
TIER_VISION_CHEAP = "vision_cheap"         # → premium (no cheap vision tier yet)
TIER_VISION_STRONG = "vision_strong"       # → premium
TIER_NO_MODEL = "no_model"                 # deterministic, zero cost

# Mapping: agent abstract tier → existing Theriad tier for model selection
AGENT_TIER_MAP: dict[str, str] = {
    TIER_CHEAP_REASONING: "cheap_code",
    TIER_VISION_CHEAP:    "premium",
    TIER_VISION_STRONG:   "premium",
    TIER_NO_MODEL:        "no_model",
    # Pass-through tiers that already exist in the core registry:
    "cheap_bulk":         "cheap_bulk",
    "cheap_code":         "cheap_code",
    "strong_reasoning":   "strong_reasoning",
}

# Steps between log summarization
SUMMARIZE_EVERY_N_STEPS = 5


@dataclass
class SubtaskRouteDecision:
    subtask_type: str
    tier: str                     # abstract tier name
    resolved_tier: str            # concrete Theriad tier for model selection
    deterministic: bool = False   # True = no model call needed
    use_screenshot: bool = False  # Hint: vision path required
    reason: str = ""


def route_subtask(
    subtask_type: str,
    *,
    policy: str = POLICY_BALANCED,
    # Context hints that influence escalation:
    confidence: float = 1.0,
    risk_level: str = "low",
    prior_failures: int = 0,
    page_is_canvas_heavy: bool = False,
    dom_was_insufficient: bool = False,
    high_stakes_domain: bool = False,
    complex_comparison: bool = False,
) -> SubtaskRouteDecision:
    """
    Return the routing decision for a single browser-agent subtask.

    Rules follow the spec:
    - observe_dom: cheap deterministic first, model only if needed
    - observe_screenshot: vision_cheap only when DOM insufficient / canvas page
    - extract_data: structured → cheap_code, prose → cheap_bulk
    - plan_next_action: cheap_reasoning; escalate → strong_reasoning on risk/failure
    - validate_action: no_model first; fallback → cheap_reasoning if ambiguous
    - summarize_log: cheap_bulk
    - final_report: cheap_bulk; escalate → strong_reasoning for complex tasks
    """
    aggressive = policy == POLICY_AGGRESSIVE_SAVINGS
    quality = policy == POLICY_QUALITY_FIRST

    if subtask_type == SUBTASK_OBSERVE_DOM:
        # Always try deterministic extraction first
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier="cheap_code",
            resolved_tier=AGENT_TIER_MAP["cheap_code"],
            deterministic=True,
            reason="DOM observation uses deterministic browser extraction.",
        )

    elif subtask_type == SUBTASK_OBSERVE_SCREENSHOT:
        # Only called when DOM is insufficient; default_enabled=False
        tier = TIER_VISION_CHEAP if not quality else TIER_VISION_STRONG
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier=tier,
            resolved_tier=AGENT_TIER_MAP[tier],
            use_screenshot=True,
            reason="Screenshot observation: DOM was insufficient or canvas-heavy page.",
        )

    elif subtask_type == SUBTASK_EXTRACT_DATA:
        # Structured → cheap_code; prose → cheap_bulk
        if quality:
            tier = "strong_reasoning"
        else:
            tier = "cheap_code"
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier=tier,
            resolved_tier=AGENT_TIER_MAP[tier],
            reason="Data extraction: structured prefers cheap_code.",
        )

    elif subtask_type == SUBTASK_PLAN_NEXT_ACTION:
        # Escalate to strong_reasoning when: risk, failure, or quality policy
        should_escalate = (
            quality
            or prior_failures > 0
            or risk_level in ("high", "critical")
            or high_stakes_domain
            or complex_comparison
            or confidence < 0.5
        )
        tier = "strong_reasoning" if should_escalate else TIER_CHEAP_REASONING
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier=tier,
            resolved_tier=AGENT_TIER_MAP[tier],
            reason=(
                f"Action planning escalated to strong_reasoning (risk={risk_level}, "
                f"failures={prior_failures})."
                if should_escalate
                else "Action planning uses cheap_reasoning."
            ),
        )

    elif subtask_type == SUBTASK_VALIDATE_ACTION:
        # Deterministic policy check first; no model unless ambiguous
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier=TIER_NO_MODEL,
            resolved_tier="no_model",
            deterministic=True,
            reason="Action validation uses deterministic policy rules (no model).",
        )

    elif subtask_type == SUBTASK_SUMMARIZE_LOG:
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier="cheap_bulk",
            resolved_tier=AGENT_TIER_MAP["cheap_bulk"],
            reason=f"Log summarization uses cheap_bulk every {SUMMARIZE_EVERY_N_STEPS} steps.",
        )

    elif subtask_type == SUBTASK_FINAL_REPORT:
        should_escalate = (
            quality
            or high_stakes_domain
            or complex_comparison
            or prior_failures > 0
        )
        tier = "strong_reasoning" if should_escalate else "cheap_bulk"
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier=tier,
            resolved_tier=AGENT_TIER_MAP[tier],
            reason=(
                "Final report escalated to strong_reasoning."
                if should_escalate
                else "Final report uses cheap_bulk."
            ),
        )

    elif subtask_type == SUBTASK_ORCHESTRATE:
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier="premium",
            resolved_tier="premium",
            reason="High-level plan orchestration uses premium model (GLM 5.2).",
        )

    elif subtask_type == SUBTASK_EVALUATE_GOAL:
        return SubtaskRouteDecision(
            subtask_type=subtask_type,
            tier="strong_reasoning",
            resolved_tier=AGENT_TIER_MAP["strong_reasoning"],
            reason="Goal evaluation uses strong_reasoning (free model).",
        )

    # Unknown subtask → cheap_bulk safe default
    return SubtaskRouteDecision(
        subtask_type=subtask_type,
        tier="cheap_bulk",
        resolved_tier="cheap_bulk",
        reason=f"Unknown subtask type '{subtask_type}'; defaulting to cheap_bulk.",
    )


def should_summarize(steps_taken: int) -> bool:
    """True when the action log should be compressed."""
    return steps_taken > 0 and steps_taken % SUMMARIZE_EVERY_N_STEPS == 0
