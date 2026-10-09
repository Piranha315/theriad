"""Tests for theriad.workflow.cost_simulator.WorkflowCostSimulator."""
from __future__ import annotations

import pytest

from theriad.workflow.cost_simulator import WorkflowCostSimulator
from theriad.workflow.planner import WorkflowPlanner
from theriad.workflow.types import ExecutionStep, WorkflowDetection


def _make_detection() -> WorkflowDetection:
    return WorkflowDetection(
        is_workflow=True,
        confidence=0.90,
        workflow_type="software_build",
        decision_source="rule_guardrail",
        reason="test",
        signals=[],
    )


def _full_plan(policy: str = "balanced") -> list[ExecutionStep]:
    return WorkflowPlanner().plan(_make_detection(), "Build a full app.", policy=policy)


class TestWorkflowCostSimulator:

    def setup_method(self):
        self.sim = WorkflowCostSimulator()

    def test_simulate_returns_dict(self):
        result = self.sim.simulate(_full_plan())
        assert isinstance(result, dict)

    def test_result_keys_present(self):
        result = self.sim.simulate(_full_plan())
        expected = {"step_costs", "total_cost_usd", "baseline_cost_usd", "savings_pct", "latency_ms"}
        assert expected.issubset(result.keys())

    def test_step_costs_length_matches_steps(self):
        steps = _full_plan()
        result = self.sim.simulate(steps)
        assert len(result["step_costs"]) == len(steps)

    def test_no_model_step_costs_zero(self):
        steps = _full_plan()
        result = self.sim.simulate(steps)
        for sc in result["step_costs"]:
            if sc["routing_tier"] == "no_model":
                assert sc["estimated_cost_usd"] == 0.0, (
                    f"no_model step {sc['name']!r} should cost $0"
                )

    def test_total_cost_is_sum_of_step_costs(self):
        steps = _full_plan()
        result = self.sim.simulate(steps)
        total = sum(sc["estimated_cost_usd"] for sc in result["step_costs"])
        assert abs(result["total_cost_usd"] - round(total, 6)) < 1e-6

    def test_baseline_cost_positive_for_known_model(self):
        result = self.sim.simulate(_full_plan(), baseline="openai/gpt-4o")
        # gpt-4o is in PricingTable — baseline should be > 0
        assert result["baseline_cost_usd"] > 0

    def test_savings_pct_reasonable(self):
        result = self.sim.simulate(_full_plan(), baseline="openai/gpt-4o")
        # Savings can be negative if theriad costs more, but must be a float
        assert isinstance(result["savings_pct"], float)

    def test_latency_ms_positive(self):
        result = self.sim.simulate(_full_plan())
        assert result["latency_ms"] > 0

    def test_each_step_cost_has_required_keys(self):
        result = self.sim.simulate(_full_plan())
        required = {
            "name", "task_type", "routing_tier", "model",
            "uses_model", "estimated_cost_usd",
            "estimated_input_tokens", "estimated_output_tokens", "estimated_latency_ms",
        }
        for sc in result["step_costs"]:
            assert required.issubset(sc.keys()), f"step {sc.get('name')!r} missing keys"

    def test_aggressive_savings_costs_less_within_each_tier(self):
        """The policy invariant, asserted where it actually holds.

        This previously compared whole *plans*: the planner assigns richer tiers
        under quality_first, so the plan-level comparison assumed the tier ladder
        was monotone in cost. It no longer is. `premium` leads with
        deepseek-v4-flash-0731 at $0.14/$0.28, which is 5.4x cheaper per token
        than `cheap_code` on qwen3.6-plus at $0.325/$1.95 — so planning a step
        into a *higher* tier can now cost less, and the plan-level assertion
        fails for a reason that has nothing to do with the policies.

        The meaningful invariant is per tier: given the same candidates,
        aggressive_savings never selects a costlier model than quality_first.
        """
        from theriad.models.policy import PolicyResolver
        from theriad.models.registry import DEFAULT_REGISTRY

        resolver = PolicyResolver()
        for tier in DEFAULT_REGISTRY.all_tiers():
            agg = resolver.resolve(tier, "aggressive_savings").selected
            qf = resolver.resolve(tier, "quality_first").selected
            cost = lambda c: (2500 * (c.input_cost_per_million or 0)
                              + 2500 * (c.output_cost_per_million or 0)) / 1e6
            assert cost(agg) <= cost(qf), (
                f"{tier}: aggressive_savings picked {agg.model} (${cost(agg):.5f}) "
                f"over quality_first's {qf.model} (${cost(qf):.5f})"
            )

    def test_empty_step_list_returns_zeros(self):
        result = self.sim.simulate([])
        assert result["total_cost_usd"] == 0.0
        assert result["latency_ms"] == 0

    def test_unknown_baseline_returns_zero_baseline_cost(self):
        result = self.sim.simulate(_full_plan(), baseline="nonexistent/unknown-model-xyz")
        assert result["baseline_cost_usd"] == 0.0

    def test_no_model_step_directly(self):
        """Tool step should always be zero cost regardless of policy."""
        step = ExecutionStep(
            name="validation",
            task_type="tool",
            description="Run tests",
            input_refs=["implementation_artifacts"],
            output_key="validation_result",
            routing_tier="no_model",
            uses_model=False,
            tool_name="pytest",
        )
        result = self.sim.simulate([step])
        assert result["total_cost_usd"] == 0.0
        assert result["step_costs"][0]["estimated_cost_usd"] == 0.0
