"""
theriad.workflow — Workflow-level cost optimization (v2).

Detects multi-step prompts, produces linear execution plans,
and estimates per-step cost — all offline, no provider calls.

Public API:
    from theriad.workflow.detector import WorkflowDetector
    from theriad.workflow.planner import WorkflowPlanner
    from theriad.workflow.cost_simulator import WorkflowCostSimulator
    from theriad.workflow.types import ExecutionStep, ExecutionPlan, WorkflowDetection
"""
