"""Outcome grading for Router V2 study responses.

Deterministic checks are authoritative.  Pairwise judges can fill an explicitly
open-ended label, but they can never overwrite a JSON, benchmark-answer, test,
or trusted-state result.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from theriad.benchmark.types import BenchmarkPrompt
from theriad.quality.objective_verifier import verify_objective_response

from .study import StudyPrompt
from .study_graders import verify_trusted_result

EVALUATOR_VERSION = "router-v2-grader-2026-08-24-v1"


@dataclass(frozen=True)
class GradeResult:
    label: bool | None
    status: str
    method: str
    evaluator_version: str = EVALUATOR_VERSION
    details: str = ""
    deterministic_label: bool | None = None
    judge_label: bool | None = None
    disagreement: bool = False
    grader_id: str | None = None
    evidence_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status,
            "method": self.method,
            "evaluator_version": self.evaluator_version,
            "details": self.details,
            "deterministic_label": self.deterministic_label,
            "judge_label": self.judge_label,
            "disagreement": self.disagreement,
            "grader_id": self.grader_id,
            "evidence_sha256": self.evidence_sha256,
        }


def _benchmark_grade(prompt: StudyPrompt, output: str) -> GradeResult:
    grader = prompt.grader
    answer = grader.get("answer")
    answer_type = grader.get("answer_type")
    expected_properties = grader.get("expected_properties", [])
    if answer is None and not expected_properties:
        return GradeResult(
            None,
            "pending",
            "benchmark_missing_answer",
            details="benchmark row has no executable answer or property checks",
        )
    item = BenchmarkPrompt(
        id=prompt.prompt_id,
        category=prompt.domain,
        prompt=prompt.prompt,
        answer=str(answer) if answer is not None else None,
        answer_type=answer_type,
        expected_properties=[str(value) for value in expected_properties],
        max_output_tokens=prompt.max_output_tokens,
    )
    label = item.quality_check(output)
    return GradeResult(
        label,
        "graded",
        "deterministic_benchmark",
        details="benchmark answer/property check",
        deterministic_label=label,
    )


def grade_response(
    prompt: StudyPrompt,
    output: str,
    *,
    external_signals: Iterable[tuple[str, bool, str]] = (),
    trusted_result: Mapping[str, Any] | None = None,
) -> GradeResult:
    """Grade one response without trusting model-reported confidence."""

    if prompt.label_type == "pairwise":
        return GradeResult(
            None,
            "pending_judge",
            "pairwise_pending",
            details="open-ended pairwise label requires a separately versioned judge",
        )

    grader = prompt.grader
    schema = grader.get("json_schema")
    signals = tuple(external_signals)
    if prompt.label_type == "tool_state" and not signals:
        if trusted_result is not None:
            verified = verify_trusted_result(
                grader,
                prompt_id=prompt.prompt_id,
                result=trusted_result,
            )
            return GradeResult(
                bool(verified["label"]),
                "graded",
                "trusted_executor",
                evaluator_version=str(verified["evaluator_version"]),
                details="cryptographically identified trusted result",
                deterministic_label=bool(verified["label"]),
                grader_id=str(verified["grader_id"]),
                evidence_sha256=str(verified["evidence_sha256"]),
            )
        return GradeResult(
            None,
            "pending_external_state",
            "trusted_state_pending",
            details="tool or agent-state labels require a trusted executor signal",
        )
    objective = verify_objective_response(
        prompt.prompt,
        output,
        expected_json_schema=schema if isinstance(schema, dict) else None,
        external_signals=signals,
    )
    if objective is not None:
        label = bool(objective.passed)
        return GradeResult(
            label,
            "graded",
            "deterministic_objective",
            details=objective.details,
            deterministic_label=label,
        )

    if "answer" in grader or grader.get("expected_properties"):
        return _benchmark_grade(prompt, output)

    return GradeResult(
        None,
        "pending",
        "no_executable_grader",
        details="no deterministic acceptance contract was supplied",
    )


@dataclass(frozen=True)
class PairwisePresentation:
    prompt_id: str
    left_answer: str
    right_answer: str
    left_is_reference: bool
    position_seed: str
    judge_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "left_answer": self.left_answer,
            "right_answer": self.right_answer,
            "left_is_reference": self.left_is_reference,
            "position_seed": self.position_seed,
            "judge_version": self.judge_version,
        }


def randomize_pairwise_positions(
    prompt_id: str,
    reference_answer: str,
    challenger_answer: str,
    *,
    judge_version: str,
) -> PairwisePresentation:
    """Randomize answer positions deterministically for resumable judging."""

    seed = hashlib.sha256(f"{judge_version}:{prompt_id}".encode()).hexdigest()
    left_is_reference = int(seed[:2], 16) % 2 == 0
    return PairwisePresentation(
        prompt_id=prompt_id,
        left_answer=reference_answer if left_is_reference else challenger_answer,
        right_answer=challenger_answer if left_is_reference else reference_answer,
        left_is_reference=left_is_reference,
        position_seed=seed,
        judge_version=judge_version,
    )


def merge_deterministic_and_judge(
    deterministic: GradeResult | None,
    judge_label: bool | None,
    *,
    judge_version: str,
) -> GradeResult:
    """Keep deterministic evidence authoritative and surface disagreements."""

    if deterministic is not None and deterministic.deterministic_label is not None:
        disagreement = judge_label is not None and judge_label != deterministic.deterministic_label
        return GradeResult(
            label=deterministic.deterministic_label,
            status="graded" if not disagreement else "graded_disagreement",
            method=deterministic.method,
            evaluator_version=deterministic.evaluator_version,
            details=(
                f"judge={judge_version}; deterministic evidence retained"
                if disagreement
                else deterministic.details
            ),
            deterministic_label=deterministic.deterministic_label,
            judge_label=judge_label,
            disagreement=disagreement,
        )
    if judge_label is None:
        return deterministic or GradeResult(None, "pending_judge", "pairwise_pending")
    return GradeResult(
        label=bool(judge_label),
        status="graded",
        method="pairwise_judge",
        evaluator_version=judge_version,
        details="open-ended judge label",
        judge_label=bool(judge_label),
    )


def grade_mix(rows: Iterable[StudyPrompt]) -> dict[str, Any]:
    """Summarize label composition for the corpus promotion report."""

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.label_type] = counts.get(row.label_type, 0) + 1
    total = max(sum(counts.values()), 1)
    return {
        "counts": counts,
        "fractions": {key: value / total for key, value in counts.items()},
        "pairwise_at_most_10_percent": counts.get("pairwise", 0) / total <= 0.10,
    }
