"""
WorkflowCostSimulator: offline cost estimation for workflow execution plans.

Design:
- Fully offline — no provider calls, no API key required.
- Uses ModelCandidate pricing from the model registry directly (not PricingTable)
  so that newly added models with registry cost data are priced correctly.
- Baseline cost uses PricingTable for the canonical premium comparison model.
- "no_model" steps always cost $0.00.
- "code_agent" is a virtual tier that maps to "code_strong" in the v1 registry.
- Token estimates are per task_type (heuristic averages); not per-prompt.
- Returns a plain dict for easy JSON serialization in the API layer.
- Never raises: pricing errors are caught and step cost defaults to 0.0.
"""
from __future__ import annotations

from .types import ExecutionStep
from theriad.models.registry import DEFAULT_BASELINE_MODEL

# ── Workflow tier → v1 registry tier ─────────────────────────────────────────

_WORKFLOW_TO_V1_TIER: dict[str, str | None] = {
    "no_model":        None,           # zero cost, no model
    "code_agent":      "code_strong",  # maps to DeepSeek V4 Flash 0731
    "free_or_floor":   "free_or_floor",
    "cheap_bulk":      "cheap_bulk",
    "cheap_code":      "cheap_code",
    "code_strong":     "code_strong",
    "strong_reasoning":"strong_reasoning",
    "premium":         "premium",
}

# ── Per-task-type token heuristics (input, output) ────────────────────────────

_TASK_TOKEN_ESTIMATES: dict[str, dict[str, int]] = {
    "planning":      {"input": 1200, "output": 600},
    "architecture":  {"input": 1600, "output": 900},
    "coding":        {"input": 2500, "output": 2500},
    "debugging":     {"input": 2000, "output": 1200},
    "summarization": {"input": 1200, "output": 400},
    "documentation": {"input": 1200, "output": 800},
    "tool":          {"input": 0,    "output": 0},
    "validation":    {"input": 0,    "output": 0},
    "extraction":    {"input": 1200, "output": 400},
    "translation":   {"input": 1200, "output": 400},
    "rewriting":     {"input": 1200, "output": 600},
    "reasoning":     {"input": 1600, "output": 900},
}

# ── Per-task-type latency heuristics (ms) ─────────────────────────────────────

_TASK_LATENCY_MS: dict[str, int] = {
    "planning":      4000,
    "architecture":  5000,
    "coding":        8000,
    "debugging":     6000,
    "summarization": 2000,
    "documentation": 3000,
    "tool":          500,
    "validation":    500,
    "extraction":    2000,
    "translation":   2000,
    "rewriting":     2000,
    "reasoning":     5000,
}

_DEFAULT_TOKENS = {"input": 1000, "output": 400}
_DEFAULT_LATENCY = 2000


class WorkflowCostSimulator:
    """
    Estimates per-step and aggregate cost for a list of ExecutionSteps.

    Usage:
        simulator = WorkflowCostSimulator()
        result = simulator.simulate(steps, policy="balanced", baseline=DEFAULT_BASELINE_MODEL)
    """

    def simulate(
        self,
        steps: list[ExecutionStep],
        policy: str = "balanced",
        baseline: str = DEFAULT_BASELINE_MODEL,
        speed_preference: int | None = None,
        quality_preference: int | None = None,
    ) -> dict:
        """
        Estimate cost, savings, and latency for a workflow plan.

        Args:
            steps:    List of ExecutionStep from WorkflowPlanner.plan().
            policy:   Cost/quality policy forwarded to PolicyResolver.
            baseline: Model slug for the "naive single-call" cost comparison.

        Returns:
            Dict with: step_costs, total_cost_usd, baseline_cost_usd,
                       savings_pct, latency_ms.
        """
        from theriad.accounting.pricing import PricingTable, estimate_cost
        from theriad.models.policy import PolicyResolver

        resolver = PolicyResolver()
        pt = PricingTable()

        step_costs = []
        total_theriad_cost = 0.0
        total_baseline_input = 0
        total_baseline_output = 0
        total_latency = 0

        for step in steps:
            token_est = _TASK_TOKEN_ESTIMATES.get(step.task_type, _DEFAULT_TOKENS)
            in_tokens = token_est["input"]
            out_tokens = token_est["output"]
            latency_ms = _TASK_LATENCY_MS.get(step.task_type, _DEFAULT_LATENCY)

            step_cost, model_used = self._estimate_step_cost(
                step, policy, resolver, in_tokens, out_tokens,
                speed_preference=speed_preference,
                quality_preference=quality_preference,
            )

            total_theriad_cost += step_cost
            total_baseline_input += in_tokens
            total_baseline_output += out_tokens
            total_latency += latency_ms

            step_costs.append({
                "name": step.name,
                "task_type": step.task_type,
                "routing_tier": step.routing_tier,
                "model": model_used,
                "uses_model": step.uses_model,
                "estimated_cost_usd": round(step_cost, 8),
                "estimated_input_tokens": in_tokens,
                "estimated_output_tokens": out_tokens,
                "estimated_latency_ms": latency_ms,
            })

        # Baseline: naive single call at baseline model for the combined token budget
        try:
            baseline_cost = estimate_cost(baseline, total_baseline_input, total_baseline_output, pt) or 0.0
        except Exception:
            baseline_cost = 0.0

        savings_pct = (
            (baseline_cost - total_theriad_cost) / baseline_cost * 100
            if baseline_cost > 0 else 0.0
        )

        return {
            "step_costs": step_costs,
            "total_cost_usd": round(total_theriad_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "savings_pct": round(savings_pct, 1),
            "latency_ms": total_latency,
        }

    @staticmethod
    def _estimate_step_cost(
        step: ExecutionStep,
        policy: str,
        resolver,
        in_tokens: int,
        out_tokens: int,
        speed_preference: int | None = None,
        quality_preference: int | None = None,
    ) -> tuple[float, str]:
        """Return (cost_usd, model_slug) for a single step."""
        if not step.uses_model or step.routing_tier == "no_model":
            return 0.0, "no_model"

        v1_tier = _WORKFLOW_TO_V1_TIER.get(step.routing_tier, step.routing_tier)
        if v1_tier is None:
            return 0.0, "no_model"

        try:
            resolution = resolver.resolve(
                v1_tier,
                policy,
                message=f"{step.task_type}: {step.description}",
                speed_preference=speed_preference,
                quality_preference=quality_preference,
            )
            candidate = resolution.selected
            inp_cost = (candidate.input_cost_per_million or 0.0) * in_tokens / 1_000_000
            out_cost = (candidate.output_cost_per_million or 0.0) * out_tokens / 1_000_000
            return round(inp_cost + out_cost, 8), candidate.model
        except Exception:
            return 0.0, "unknown"
