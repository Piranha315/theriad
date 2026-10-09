"""
WorkflowPlanner: generates linear execution plans from workflow templates.

Design:
- Template-based only — no autonomous agent, no dynamic DAG generation.
- Only one template exists: software_build (6 steps).
- StepRouter is used to assign routing tiers per policy so that
  policy choices propagate through the full plan.
- WorkflowPlanner is stateless; plan() creates new step instances each call.
- future_templates: data_pipeline, research_report, etc. would go here.
"""
from __future__ import annotations

from .step_router import StepRouter
from .types import ExecutionStep, WorkflowDetection


class WorkflowPlanner:
    """
    Produces a linear list of ExecutionSteps for a detected workflow.

    Usage:
        planner = WorkflowPlanner()
        steps = planner.plan(detection, prompt, policy="balanced")
    """

    def __init__(self) -> None:
        self._step_router = StepRouter()

    def plan(
        self,
        detection: WorkflowDetection,
        prompt: str,
        policy: str = "balanced",
    ) -> list[ExecutionStep]:
        """
        Build an execution plan for the detected workflow type.

        Args:
            detection: WorkflowDetection from WorkflowDetector.detect().
            prompt:    Original user prompt (kept for future context injection).
            policy:    Cost/quality policy; forwarded to StepRouter.

        Returns:
            Ordered list of ExecutionStep objects.
        """
        wf_type = detection.workflow_type or "software_build"

        if wf_type == "software_build":
            steps = self._software_build_template(prompt)
        else:
            # Unknown template: fall back to software_build
            steps = self._software_build_template(prompt)

        # Apply policy-aware tier assignment
        for step in steps:
            step.routing_tier = self._step_router.route(step.task_type, policy)
            step.uses_model = step.routing_tier != "no_model"

        return steps

    def _software_build_template(self, prompt: str) -> list[ExecutionStep]:
        """
        6-step linear plan for software construction tasks.

        Step order: requirements → architecture → implementation →
                    validation → repair → summary.
        """
        return [
            ExecutionStep(
                name="requirements_analysis",
                task_type="planning",
                description=(
                    "Analyze the user's request and produce a structured requirements document "
                    "covering scope, constraints, acceptance criteria, and open questions."
                ),
                input_refs=["user_prompt"],
                output_key="requirements_doc",
                routing_tier="",  # set by caller via StepRouter
                validation="required_sections_present",
                fallback_tier="premium",
                max_retries=0,
                uses_model=True,
                tool_name=None,
            ),
            ExecutionStep(
                name="architecture_design",
                task_type="architecture",
                description=(
                    "Design the high-level system architecture based on the requirements document. "
                    "Include component diagram, data flow, and technology choices."
                ),
                input_refs=["requirements_doc"],
                output_key="architecture_doc",
                routing_tier="",
                validation="architecture_checklist",
                fallback_tier="premium",
                max_retries=0,
                uses_model=True,
                tool_name=None,
            ),
            ExecutionStep(
                name="implementation",
                task_type="coding",
                description=(
                    "Implement code artifacts as specified in the architecture document. "
                    "Produce complete, runnable files with inline documentation."
                ),
                input_refs=["requirements_doc", "architecture_doc"],
                output_key="implementation_artifacts",
                routing_tier="",
                validation="generated_artifacts_present",
                fallback_tier="code_agent",
                max_retries=0,
                uses_model=True,
                tool_name=None,
            ),
            ExecutionStep(
                name="validation",
                task_type="tool",
                description=(
                    "Run automated checks: tests, linting, type-checking, and formatting. "
                    "No model inference — executes local tools only."
                ),
                input_refs=["implementation_artifacts"],
                output_key="validation_result",
                routing_tier="",
                validation="tool_exit_code_zero",
                fallback_tier="code_agent",
                max_retries=0,
                uses_model=False,  # overridden: always False for tool steps
                tool_name="auto_detect_validation",
            ),
            ExecutionStep(
                name="repair",
                task_type="debugging",
                description=(
                    "Fix any failures reported by the validation step. "
                    "Patch code to make all checks pass."
                ),
                input_refs=["implementation_artifacts", "validation_result"],
                output_key="repaired_artifacts",
                routing_tier="",
                validation="validation_passes_after_repair",
                fallback_tier="strong_reasoning",
                max_retries=1,
                uses_model=True,
                tool_name=None,
            ),
            ExecutionStep(
                name="final_summary",
                task_type="summarization",
                description=(
                    "Write a concise summary of what was built, key decisions made, "
                    "and any known limitations or follow-up tasks."
                ),
                input_refs=["repaired_artifacts", "requirements_doc"],
                output_key="final_summary",
                routing_tier="",
                validation="required_summary_sections_present",
                fallback_tier="cheap_code",
                max_retries=0,
                uses_model=True,
                tool_name=None,
            ),
        ]
