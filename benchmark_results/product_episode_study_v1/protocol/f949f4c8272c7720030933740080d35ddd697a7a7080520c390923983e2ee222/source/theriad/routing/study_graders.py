"""Cryptographically identified trusted-result adapters for Router V2.

The paid collector never invents tool or agent-state labels.  A separate local
executor may submit a result envelope; this module verifies the pinned executor
identity and SHA-256 of the evidence before the result can be treated as a
grade.  A grader name by itself is intentionally insufficient.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

TRUSTED_RESULT_SCHEMA_VERSION = 1


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_trusted_result(
    grader: Mapping[str, Any],
    *,
    prompt_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one executor envelope and return its immutable grade details."""

    if not isinstance(result, Mapping):
        raise ValueError("trusted grader result must be an object")
    if result.get("schema_version") != TRUSTED_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported trusted grader result schema")
    if str(result.get("prompt_id") or "") != prompt_id:
        raise ValueError("trusted grader result prompt_id mismatch")
    if not isinstance(result.get("result"), bool):
        raise ValueError("trusted grader result must contain a boolean result")
    expected_executor = str(grader.get("executor_id") or "")
    expected_version = str(grader.get("executor_version") or "")
    if str(result.get("executor_id") or "") != expected_executor:
        raise ValueError("trusted grader executor_id mismatch")
    if str(result.get("executor_version") or "") != expected_version:
        raise ValueError("trusted grader executor_version mismatch")
    evidence = result.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("trusted grader result needs an evidence object")
    evidence_hash = str(result.get("evidence_sha256") or "").strip().lower()
    if evidence_hash != canonical_hash(evidence):
        raise ValueError("trusted grader evidence SHA-256 mismatch")
    return {
        "label": bool(result["result"]),
        "method": "trusted_executor",
        "grader_id": expected_executor,
        "evaluator_version": expected_version,
        "evidence_sha256": evidence_hash,
        "evidence_policy": str(grader.get("evidence_policy") or ""),
    }


def validate_recorded_grade(
    grader: Mapping[str, Any], grade: Mapping[str, Any]
) -> None:
    """Fail closed when finalization sees an unproven external grade."""

    if not isinstance(grade, Mapping):
        raise ValueError("recorded grade must be an object")
    kind = str(grader.get("kind") or "")
    if kind.endswith("trusted_result"):
        if grade.get("method") != "trusted_executor":
            raise ValueError("trusted-result row lacks trusted_executor evidence")
        if grade.get("grader_id") != grader.get("executor_id"):
            raise ValueError("recorded trusted grader identity mismatch")
        if grade.get("evaluator_version") != grader.get("executor_version"):
            raise ValueError("recorded trusted grader version mismatch")
        evidence_hash = str(grade.get("evidence_sha256") or "")
        if len(evidence_hash) != 64:
            raise ValueError("recorded trusted grade lacks evidence_sha256")


def validate_pairwise_contract(grader: Mapping[str, Any]) -> None:
    """Validate the frozen pairwise judge identity and sampling contract."""

    required = (
        "judge_id",
        "judge_version",
        "criteria_version",
        "position_randomization",
        "sampling_settings",
        "tie_policy",
        "abstention_policy",
        "disagreement_review_policy",
        "judging_cost_status",
    )
    if any(not str(grader.get(key) or "").strip() for key in required):
        raise ValueError("pairwise grader contract is incomplete")
    settings = grader.get("sampling_settings")
    if not isinstance(settings, Mapping):
        raise ValueError("pairwise sampling_settings must be an object")
    if settings.get("temperature") != 0.0:
        raise ValueError("pairwise judge temperature must be frozen at 0")
    if settings.get("top_p") != 1.0:
        raise ValueError("pairwise judge top_p must be frozen at 1")
    status = str(grader.get("judging_cost_status") or "")
    if status not in {"priced", "postponed_unpriced"}:
        raise ValueError("pairwise judging_cost_status must be priced or postponed_unpriced")
    expected = grader.get("expected_judging_cost_usd")
    worst = grader.get("worst_case_judging_cost_usd")
    if status == "priced":
        if not isinstance(expected, int | float) or not isinstance(worst, int | float):
            raise ValueError("priced pairwise contract needs expected and worst-case judging costs")
        if float(expected) < 0 or float(worst) < float(expected):
            raise ValueError("pairwise judging costs are not monotonic")
    elif expected is not None or worst is not None:
        raise ValueError(
            "postponed pairwise grading must keep expected and worst-case costs explicitly unpriced"
        )
