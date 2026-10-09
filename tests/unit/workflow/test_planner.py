"""Tests for theriad.workflow.planner.WorkflowPlanner."""
from __future__ import annotations

import pytest

from theriad.workflow.planner import WorkflowPlanner
from theriad.workflow.types import ExecutionStep, WorkflowDetection


def _make_detection(workflow_type: str = "software_build") -> WorkflowDetection:
    return WorkflowDetection(
        is_workflow=True,
        confidence=0.90,
        workflow_type=workflow_type,
        decision_source="rule_guardrail",
        reason="test",
        signals=[],
    )


class TestWorkflowPlanner:

    def setup_method(self):
        self.planner = WorkflowPlanner()

    def test_returns_six_steps(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        assert len(steps) == 6

    def test_all_steps_are_execution_step(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        assert all(isinstance(s, ExecutionStep) for s in steps)

    def test_step_names_are_correct(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        names = [s.name for s in steps]
        assert names == [
            "requirements_analysis",
            "architecture_design",
            "implementation",
            "validation",
            "repair",
            "final_summary",
        ]

    def test_validation_step_uses_no_model(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        val = next(s for s in steps if s.name == "validation")
        assert val.uses_model is False
        assert val.routing_tier == "no_model"
        assert val.tool_name is not None

    def test_model_steps_use_model(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        model_steps = [s for s in steps if s.name != "validation"]
        assert all(s.uses_model for s in model_steps)

    def test_all_routing_tiers_set(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        for step in steps:
            assert step.routing_tier, f"step {step.name!r} has empty routing_tier"

    def test_output_keys_are_unique(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        keys = [s.output_key for s in steps]
        assert len(keys) == len(set(keys)), "output_key values should be unique"

    def test_input_refs_reference_prior_outputs(self):
        steps = self.planner.plan(_make_detection(), "Build a React app.")
        available = {"user_prompt"}
        for step in steps:
            for ref in step.input_refs:
                assert ref in available, (
                    f"step {step.name!r} references {ref!r} which is not yet available"
                )
            available.add(step.output_key)

    # ── Policy-specific tier checks ───────────────────────────────────────────

    def test_balanced_requirements_uses_strong_reasoning(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="balanced")
        req = next(s for s in steps if s.name == "requirements_analysis")
        assert req.routing_tier == "strong_reasoning"

    def test_balanced_implementation_uses_cheap_code(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="balanced")
        impl = next(s for s in steps if s.name == "implementation")
        assert impl.routing_tier == "cheap_code"

    def test_balanced_repair_uses_code_agent(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="balanced")
        repair = next(s for s in steps if s.name == "repair")
        assert repair.routing_tier == "code_agent"

    def test_balanced_summary_uses_cheap_bulk(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="balanced")
        summary = next(s for s in steps if s.name == "final_summary")
        assert summary.routing_tier == "cheap_bulk"

    def test_aggressive_savings_downgrades_planning(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="aggressive_savings")
        req = next(s for s in steps if s.name == "requirements_analysis")
        # aggressive_savings maps planning → cheap_code
        assert req.routing_tier == "cheap_code"

    def test_quality_first_upgrades_planning(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="quality_first")
        req = next(s for s in steps if s.name == "requirements_analysis")
        assert req.routing_tier == "premium"

    def test_unknown_workflow_type_falls_back_to_software_build(self):
        detection = _make_detection(workflow_type="unknown_future_type")
        steps = self.planner.plan(detection, "Do something complex.", policy="balanced")
        assert len(steps) == 6

    def test_fallback_tiers_set_on_model_steps(self):
        steps = self.planner.plan(_make_detection(), "Build.", policy="balanced")
        model_steps = [s for s in steps if s.uses_model]
        for step in model_steps:
            assert step.fallback_tier is not None, (
                f"model step {step.name!r} should have a fallback_tier"
            )
