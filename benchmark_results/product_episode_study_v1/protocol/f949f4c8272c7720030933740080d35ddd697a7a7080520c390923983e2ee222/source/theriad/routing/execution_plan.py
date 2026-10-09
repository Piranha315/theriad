"""Immutable, hash-bound executable plans for Router V2 study phases.

The reviewed phased plan is deliberately not executable: it contains a
planned schedule but no active calls.  This module is the only boundary that
may turn one locked phase into an executable single-phase plan.  The caller
must persist and review the resulting hash; execution never materializes a
schedule in memory from a locked plan.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .study_collector import PlannedStudyCall, StudyCollectorError, StudyPlan

EXECUTION_PLAN_SCHEMA_VERSION = 1


class ExecutionPlanError(StudyCollectorError):
    """Raised when an executable phase plan is missing or has drifted."""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_from_dict(value: dict[str, Any]) -> PlannedStudyCall:
    return PlannedStudyCall(
        prompt_id=str(value["prompt_id"]),
        model=str(value["model"]),
        split=str(value["split"]),
        max_output_tokens=int(value["max_output_tokens"]),
        maximum_cost_usd=float(value["maximum_cost_usd"]),
        idempotency_key=str(value["idempotency_key"]),
        input_tokens=int(value.get("input_tokens", 0)),
        expected_output_tokens=int(value.get("expected_output_tokens", 0)),
        expected_cost_usd=float(value.get("expected_cost_usd", 0.0)),
        grading_cost_usd=float(value.get("grading_cost_usd", 0.0)),
        task_format=str(value.get("task_format") or ""),
        adapter_id=str(value.get("adapter_id") or ""),
        grading_status=str(value.get("grading_status") or ""),
        expected_turns=int(value.get("expected_turns", 1)),
        max_turns=int(value.get("max_turns", 1)),
        potential_api_calls=int(value.get("potential_api_calls", 1)),
        expected_trajectory_tokens=int(value.get("expected_trajectory_tokens", 0)),
        max_trajectory_tokens=int(value.get("max_trajectory_tokens", 0)),
    )


def _calls_as_dict(calls: tuple[PlannedStudyCall, ...]) -> list[dict[str, Any]]:
    return [call.to_dict() for call in calls]


@dataclass(frozen=True)
class ExecutablePhasePlan:
    """A complete, single-phase plan whose hash covers every call and gate."""

    phase: str
    parent_plan_hash: str
    parent_phased_plan_hash: str | None
    corpus_hash: str
    status: str
    calls: tuple[PlannedStudyCall, ...]
    price_snapshot: dict[str, dict[str, Any]]
    cost_scenarios: dict[str, dict[str, Any]]
    max_attempts: int
    generated_at: str
    preconditions: tuple[str, ...]
    grading_cost_status: str
    executor_manifest_hash: str
    anchor_boundary_hash: str
    metadata_snapshot: dict[str, Any]
    metadata_response_sha256: str
    metadata_fetched_at: str
    execution_plan_hash: str

    def _payload(self) -> dict[str, Any]:
        """Return the exact hash input, excluding only the derived hash."""

        return {
            "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "phase": self.phase,
            "parent_plan_hash": self.parent_plan_hash,
            "parent_phased_plan_hash": self.parent_phased_plan_hash,
            "corpus_hash": self.corpus_hash,
            "status": self.status,
            "calls": _calls_as_dict(self.calls),
            "price_snapshot": self.price_snapshot,
            "cost_scenarios": self.cost_scenarios,
            "max_attempts": self.max_attempts,
            "generated_at": self.generated_at,
            "preconditions": list(self.preconditions),
            "grading_cost_status": self.grading_cost_status,
            "executor_manifest_hash": self.executor_manifest_hash,
            "anchor_boundary_hash": self.anchor_boundary_hash,
            "metadata_snapshot": self.metadata_snapshot,
            "metadata_response_sha256": self.metadata_response_sha256,
            "metadata_fetched_at": self.metadata_fetched_at,
            "call_count": self.call_count,
            "task_model_runs": self.call_count,
            "potential_api_call_count": self.potential_api_call_count,
            "maximum_cost_usd": self.maximum_cost_usd,
            "expected_cost_usd": self.expected_cost_usd,
            "grading_cost_usd": self.grading_cost_usd,
            "worst_case_total_cost_usd": self.maximum_cost_usd + self.grading_cost_usd,
            "planned_call_count": self.call_count,
            "planned_potential_api_call_count": self.potential_api_call_count,
            "planned_maximum_cost_usd": self.maximum_cost_usd,
            "planned_expected_cost_usd": self.expected_cost_usd,
        }

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def potential_api_call_count(self) -> int:
        return sum(max(1, call.potential_api_calls) for call in self.calls)

    @property
    def maximum_cost_usd(self) -> float:
        return sum(call.maximum_cost_usd for call in self.calls)

    @property
    def expected_cost_usd(self) -> float:
        return sum(call.expected_cost_usd for call in self.calls)

    @property
    def grading_cost_usd(self) -> float:
        return sum(call.grading_cost_usd for call in self.calls)

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload.update(
            {
                "execution_plan_hash": self.execution_plan_hash,
                "call_count": self.call_count,
                "task_model_runs": self.call_count,
                "potential_api_call_count": self.potential_api_call_count,
                "maximum_cost_usd": self.maximum_cost_usd,
                "expected_cost_usd": self.expected_cost_usd,
                "grading_cost_usd": self.grading_cost_usd,
                "worst_case_total_cost_usd": self.maximum_cost_usd + self.grading_cost_usd,
                "planned_call_count": self.call_count,
                "planned_potential_api_call_count": self.potential_api_call_count,
                "planned_maximum_cost_usd": self.maximum_cost_usd,
                "planned_expected_cost_usd": self.expected_cost_usd,
            }
        )
        return payload

    def to_study_plan(self) -> StudyPlan:
        """Adapt the verified file for the collector without re-materializing it."""

        return StudyPlan(
            calls=self.calls,
            planned_calls=self.calls,
            price_snapshot=self.price_snapshot,
            corpus_hash=self.corpus_hash,
            max_attempts=self.max_attempts,
            generated_at=self.generated_at,
            plan_hash=self.execution_plan_hash,
            cost_scenarios=self.cost_scenarios,
            phase=self.phase,
            status=self.status,
            preconditions=self.preconditions,
            potential_call_count=self.potential_api_call_count,
            grading_cost_status=self.grading_cost_status,
            planned_call_count=self.call_count,
            planned_potential_api_call_count=self.potential_api_call_count,
            planned_maximum_cost_usd=self.maximum_cost_usd,
            planned_expected_cost_usd=self.expected_cost_usd,
        )


def materialize_execution_plan(
    locked_plan: StudyPlan,
    *,
    parent_phased_plan_hash: str | None,
    executor_manifest_hash: str,
    anchor_boundary_hash: str,
    metadata_snapshot: dict[str, Any],
    metadata_response_sha256: str,
    metadata_fetched_at: str,
) -> ExecutablePhasePlan:
    """Materialize a locked phase exactly once, provider-free."""

    if locked_plan.status != "locked_pending_provider_approval":
        raise ExecutionPlanError(
            f"only a locked phase may be materialized; got status {locked_plan.status!r}"
        )
    if locked_plan.calls:
        raise ExecutionPlanError("locked phase unexpectedly contains active calls")
    if not locked_plan.planned_calls:
        raise ExecutionPlanError("locked phase has no planned call schedule")
    if not locked_plan.plan_hash:
        raise ExecutionPlanError("locked phase has no parent plan hash")
    if not executor_manifest_hash or not anchor_boundary_hash:
        raise ExecutionPlanError("execution plan requires manifest and anchor-boundary hashes")
    if not metadata_response_sha256 or not metadata_fetched_at:
        raise ExecutionPlanError("execution plan requires a frozen metadata hash and timestamp")
    plan = ExecutablePhasePlan(
        phase=locked_plan.phase,
        parent_plan_hash=locked_plan.plan_hash,
        parent_phased_plan_hash=parent_phased_plan_hash,
        corpus_hash=locked_plan.corpus_hash,
        status="ready",
        calls=tuple(locked_plan.planned_calls),
        price_snapshot=deepcopy(locked_plan.price_snapshot),
        cost_scenarios=deepcopy(locked_plan.cost_scenarios),
        max_attempts=locked_plan.max_attempts,
        generated_at=locked_plan.generated_at,
        preconditions=locked_plan.preconditions,
        grading_cost_status=locked_plan.grading_cost_status,
        executor_manifest_hash=executor_manifest_hash,
        anchor_boundary_hash=anchor_boundary_hash,
        metadata_snapshot=deepcopy(metadata_snapshot),
        metadata_response_sha256=metadata_response_sha256,
        metadata_fetched_at=metadata_fetched_at,
        execution_plan_hash="",
    )
    return ExecutablePhasePlan(
        **{
            **plan.__dict__,
            "execution_plan_hash": canonical_hash(plan._payload()),
        }
    )


def load_execution_plan(path: str | Path) -> ExecutablePhasePlan:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionPlanError(f"executable phase plan is unreadable: {target}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != EXECUTION_PLAN_SCHEMA_VERSION:
        raise ExecutionPlanError("unsupported executable phase plan schema")
    calls_raw = raw.get("calls")
    if not isinstance(calls_raw, list):
        raise ExecutionPlanError("executable phase plan has no complete calls list")
    calls = tuple(_call_from_dict(value) for value in calls_raw if isinstance(value, dict))
    if len(calls) != len(calls_raw) or not calls:
        raise ExecutionPlanError("executable phase plan calls are malformed or empty")
    plan = ExecutablePhasePlan(
        phase=str(raw.get("phase") or ""),
        parent_plan_hash=str(raw.get("parent_plan_hash") or ""),
        parent_phased_plan_hash=(
            str(raw["parent_phased_plan_hash"])
            if raw.get("parent_phased_plan_hash") is not None
            else None
        ),
        corpus_hash=str(raw.get("corpus_hash") or ""),
        status=str(raw.get("status") or ""),
        calls=calls,
        price_snapshot=dict(raw.get("price_snapshot") or {}),
        cost_scenarios=dict(raw.get("cost_scenarios") or {}),
        max_attempts=int(raw.get("max_attempts", 1)),
        generated_at=str(raw.get("generated_at") or ""),
        preconditions=tuple(str(item) for item in raw.get("preconditions", [])),
        grading_cost_status=str(raw.get("grading_cost_status") or ""),
        executor_manifest_hash=str(raw.get("executor_manifest_hash") or ""),
        anchor_boundary_hash=str(raw.get("anchor_boundary_hash") or ""),
        metadata_snapshot=dict(raw.get("metadata_snapshot") or {}),
        metadata_response_sha256=str(raw.get("metadata_response_sha256") or ""),
        metadata_fetched_at=str(raw.get("metadata_fetched_at") or ""),
        execution_plan_hash=str(raw.get("execution_plan_hash") or ""),
    )
    if plan.status != "ready":
        raise ExecutionPlanError("executable phase plan status is not ready")
    expected = canonical_hash(plan._payload())
    if not plan.execution_plan_hash or plan.execution_plan_hash != expected:
        raise ExecutionPlanError("execution plan hash does not match its contents")
    return plan


def write_execution_plan(plan: ExecutablePhasePlan, path: str | Path) -> bytes:
    """Write once and refuse to overwrite a different executable plan."""

    target = Path(path)
    encoded = (json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise ExecutionPlanError(f"cannot read existing executable plan: {target}") from exc
        if existing != encoded:
            raise ExecutionPlanError(
                f"refusing to overwrite immutable executable plan with different contents: {target}"
            )
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
        try:
            target.chmod(0o444)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    return encoded


def assert_matches_locked_plan(
    executable: ExecutablePhasePlan,
    locked_plan: StudyPlan,
    *,
    parent_phased_plan_hash: str | None,
    corpus_hash: str,
    executor_manifest_hash: str,
    anchor_boundary_hash: str,
) -> None:
    """Reject any parent, schedule, gate, or provenance drift."""

    if locked_plan.status != "locked_pending_provider_approval" or locked_plan.calls:
        raise ExecutionPlanError("parent phase is no longer the locked, unmaterialized plan")
    checks = (
        (executable.phase == locked_plan.phase, "phase changed"),
        (executable.parent_plan_hash == locked_plan.plan_hash, "parent phase hash changed"),
        (executable.parent_phased_plan_hash == parent_phased_plan_hash, "parent phased-plan hash changed"),
        (executable.corpus_hash == corpus_hash == locked_plan.corpus_hash, "corpus hash changed"),
        (executable.status == "ready", "execution status changed"),
        (executable.calls == locked_plan.planned_calls, "executable call schedule changed"),
        (executable.price_snapshot == locked_plan.price_snapshot, "price snapshot changed"),
        (executable.executor_manifest_hash == executor_manifest_hash, "executor manifest changed"),
        (executable.anchor_boundary_hash == anchor_boundary_hash, "anchor boundary changed"),
    )
    for passed, reason in checks:
        if not passed:
            raise ExecutionPlanError(reason)


def assert_reviewed_hash(executable: ExecutablePhasePlan, reviewed_hash: str) -> None:
    """Authorize only the newly materialized executable-plan hash."""

    if str(reviewed_hash or "") != executable.execution_plan_hash:
        raise ExecutionPlanError(
            "reviewed plan hash does not match the immutable execution-plan hash"
        )


__all__ = [
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "ExecutablePhasePlan",
    "ExecutionPlanError",
    "assert_matches_locked_plan",
    "assert_reviewed_hash",
    "canonical_hash",
    "file_sha256",
    "load_execution_plan",
    "materialize_execution_plan",
    "write_execution_plan",
]
