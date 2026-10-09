"""
StepRouter: maps workflow task types to abstract routing tiers.

Design:
- Base mapping is policy-agnostic ("balanced" policy).
- aggressive_savings overrides downgrade costly task types to cheaper tiers.
- quality_first overrides upgrade planning/architecture steps to premium.
- Unknown task types fall through to "cheap_bulk" to avoid crashes.
- "no_model" and "code_agent" are virtual tiers:
    no_model   → step executes a local tool, zero model cost.
    code_agent → maps to code_strong in the v1 registry.
"""
from __future__ import annotations

_BASE_TIER: dict[str, str] = {
    "planning":      "strong_reasoning",
    "architecture":  "strong_reasoning",
    "coding":        "cheap_code",
    "debugging":     "code_agent",
    "tool":          "no_model",
    "validation":    "no_model",
    "summarization": "cheap_bulk",
    "documentation": "cheap_bulk",
    "extraction":    "cheap_bulk",
    "translation":   "cheap_bulk",
    "rewriting":     "cheap_bulk",
    "reasoning":     "strong_reasoning",
}

_AGGRESSIVE_SAVINGS_OVERRIDES: dict[str, str] = {
    "planning":     "cheap_code",
    "architecture": "cheap_code",
    "debugging":    "cheap_code",
    "reasoning":    "cheap_code",
}

_QUALITY_FIRST_OVERRIDES: dict[str, str] = {
    "planning":     "premium",
    "architecture": "premium",
    "debugging":    "strong_reasoning",
    "reasoning":    "premium",
    "coding":       "code_agent",
}


class StepRouter:
    """
    Maps a workflow step's task_type to an abstract routing tier.

    Usage:
        router = StepRouter()
        tier = router.route("planning", policy="balanced")
        # → "strong_reasoning"
    """

    def route(self, task_type: str, policy: str = "balanced") -> str:
        """
        Return the abstract routing tier for task_type under the given policy.

        Args:
            task_type: One of the task types defined in _BASE_TIER.
            policy:    "balanced", "aggressive_savings", or "quality_first".

        Returns:
            Abstract tier string.
        """
        base = _BASE_TIER.get(task_type, "cheap_bulk")

        if policy == "aggressive_savings":
            return _AGGRESSIVE_SAVINGS_OVERRIDES.get(task_type, base)

        if policy == "quality_first":
            return _QUALITY_FIRST_OVERRIDES.get(task_type, base)

        return base
