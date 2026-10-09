"""
Core data types for workflow planning.

Design:
- Pure dataclasses, no external dependencies, always serialisable via dataclasses.asdict().
- ExecutionStep is the atomic unit; WorkflowPlanner produces lists of these.
- WorkflowDetection carries the classifier output so callers can inspect why
  the detector made its decision (rule vs embedding vs default).
- ExecutionPlan wraps detection + steps + cost estimate in one JSON-friendly object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionStep:
    """One step in a linear workflow execution plan."""

    name: str
    """Short machine-readable identifier, e.g. 'requirements_analysis'."""

    task_type: str
    """Abstract task category that StepRouter uses to assign a tier.
    One of: planning, architecture, coding, debugging, tool,
            summarization, documentation, extraction, translation, rewriting."""

    description: str
    """Human-readable description of what this step should produce."""

    input_refs: list[str]
    """Keys of upstream outputs this step depends on, or ['user_prompt']."""

    output_key: str
    """Key under which this step's output is stored for downstream steps."""

    routing_tier: str
    """Abstract routing tier assigned by StepRouter.
    One of: free_or_floor, cheap_bulk, cheap_code, code_strong,
            strong_reasoning, premium, no_model, code_agent."""

    validation: str | None = None
    """Optional validation gate name (passed to future execution engine)."""

    fallback_tier: str | None = None
    """Tier to escalate to if the primary tier fails."""

    max_retries: int = 0
    """How many times the execution engine may retry before escalating."""

    uses_model: bool = True
    """False for tool/validation steps that run local code, not a model."""

    tool_name: str | None = None
    """Name of the tool to invoke for tool steps (uses_model=False)."""


@dataclass
class WorkflowDetection:
    """Output of WorkflowDetector.detect()."""

    is_workflow: bool
    """True → planner should produce a multi-step plan; False → route via v1."""

    confidence: float
    """Classifier confidence [0.0, 1.0]."""

    workflow_type: str | None
    """Template to use if is_workflow is True, e.g. 'software_build'.
    None when is_workflow is False."""

    decision_source: str
    """Which layer made the final decision: 'rule_guardrail', 'mrl_embedding', or 'default'."""

    reason: str
    """Human-readable explanation of the decision."""

    signals: list[str] = field(default_factory=list)
    """Tokens / signals that drove the decision."""

    semantic_margin: float | None = None
    """workflow_sim − single_sim from embedding layer. None when rule guardrail was final."""

    embedding_model: str | None = None
    """Model name used by the embedding layer. None when rule guardrail was final."""

    embedding_dims_used: int | None = None
    """MRL dimension that produced a decisive margin. None when rule guardrail was final."""


@dataclass
class ExecutionPlan:
    """Complete offline workflow plan produced by WorkflowPlanner + WorkflowCostSimulator."""

    original_prompt: str
    workflow_selected: bool
    reason: str
    workflow_type: str | None
    signals: list[str]
    steps: list[ExecutionStep]
    estimated_total_cost_usd: float
    baseline_estimated_cost_usd: float
    estimated_savings_pct: float
    estimated_latency_ms: int
    policy: str
    baseline: str
