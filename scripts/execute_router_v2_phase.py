#!/usr/bin/env python3
"""Preflight or execute exactly one gated Router V2 study phase.

Provider-free preflight materializes a complete immutable executable plan.  A
locked phased plan is never converted to active calls inside execution, and a
locked hash can never authorize the resulting executable file.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.core.providers.interfaces import ProviderError  # noqa: E402
from theriad.routing.execution_plan import (  # noqa: E402
    ExecutionPlanError,
    assert_matches_locked_plan,
    assert_reviewed_hash,
    file_sha256,
    load_execution_plan,
    materialize_execution_plan,
    write_execution_plan,
)
from theriad.routing.executor_manifest import (  # noqa: E402
    ExecutorManifestError,
    load_executor_manifest,
)
from theriad.routing.openrouter_metadata import (  # noqa: E402
    OpenRouterMetadataError,
    compare_metadata_to_plan,
    fetch_openrouter_metadata,
    write_metadata_snapshot,
)
from theriad.routing.study import (  # noqa: E402
    StudyCorpusError,
    corpus_checksum,
    load_jsonl,
    validate_corpus,
)
from theriad.routing.study_adapters import (  # noqa: E402
    adapter_for_prompt,
    collection_eligibility,
)
from theriad.routing.study_anchor_boundary import StudyAnchorExclusionBoundary  # noqa: E402
from theriad.routing.study_budget import StudyBudgetConfig, StudyBudgetLedger  # noqa: E402
from theriad.routing.study_collector import (  # noqa: E402
    PHASE_A,
    PHASE_B,
    PHASE_C,
    PHASE_D,
    STUDY_PHASES,
    StudyCollector,
    StudyCollectorError,
    StudyPlan,
    corpus_fingerprint,
    load_phased_plan,
    load_plan,
)
from theriad.runtime.circuit_breaker import CircuitOpenError  # noqa: E402


def _load_phase(path: Path, phase_name: str) -> tuple[StudyPlan, str, str | None]:
    """Load only the reviewed locked parent; return phase and phased hashes."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and isinstance(raw.get("phases"), Mapping):
        phased = load_phased_plan(path)
        plan = phased.phases.get(phase_name)
        if plan is None:
            raise StudyCollectorError(f"plan has no requested phase {phase_name}")
        return plan, plan.plan_hash, phased.plan_hash
    plan = load_plan(path)
    if plan.phase != phase_name:
        raise StudyCollectorError(
            f"single-phase plan is {plan.phase!r}, not requested {phase_name!r}"
        )
    return plan, plan.plan_hash, None


def _needs_executor_manifest(prompts: tuple[Any, ...]) -> bool:
    return any(adapter_for_prompt(row).adapter_id != "plain-text" for row in prompts)


def _apply_manifest_runtime(manifest: Mapping[str, Any]) -> None:
    code = ((manifest.get("runtimes") or {}).get("code_grader") or {})
    runtime = str(code.get("requested_executable") or "").strip()
    if runtime:
        os.environ["THERIAD_ROUTER_V2_CODE_GRADER_PYTHON"] = runtime


def _static_preflight(
    prompts: tuple[Any, ...],
    plan: StudyPlan,
    *,
    overlap_manifest: Path,
    executor_manifest: Path | None,
) -> dict[str, Any]:
    expected = corpus_fingerprint(prompts)
    if plan.corpus_hash != expected:
        raise StudyCollectorError("phase plan corpus hash does not match the supplied corpus")
    try:
        boundary = StudyAnchorExclusionBoundary.load(overlap_manifest)
    except (OSError, ValueError) as exc:
        raise StudyCollectorError(f"public-anchor overlap boundary is unavailable: {exc}") from exc
    if boundary.study_corpus_checksum != corpus_checksum(prompts):
        raise StudyCollectorError("public-anchor overlap boundary belongs to another corpus")
    if not boundary.manifest_hash:
        raise StudyCollectorError("public-anchor overlap boundary has no verified manifest hash")

    manifest: dict[str, Any] | None = None
    if _needs_executor_manifest(prompts):
        if executor_manifest is None:
            raise StudyCollectorError("task-family executors require --executor-manifest")
        try:
            manifest = load_executor_manifest(executor_manifest)
        except ExecutorManifestError as exc:
            raise StudyCollectorError(str(exc)) from exc
        _apply_manifest_runtime(manifest)

    prompt_by_id = {row.prompt_id: row for row in prompts}
    issues: list[dict[str, str]] = []
    calls = plan.planned_calls or plan.calls
    allowed_split = {
        PHASE_A: "train",
        PHASE_B: "train",
        PHASE_C: "validation",
        PHASE_D: "test",
    }[plan.phase]
    seen_keys: set[str] = set()
    for call in calls:
        row = prompt_by_id.get(call.prompt_id)
        if row is None:
            issues.append({"prompt_id": call.prompt_id, "reason": "missing prompt"})
            continue
        if call.split != allowed_split:
            issues.append(
                {"prompt_id": call.prompt_id, "reason": f"call is outside {allowed_split} split"}
            )
        if call.idempotency_key in seen_keys:
            issues.append(
                {"prompt_id": call.prompt_id, "reason": "duplicate idempotency key"}
            )
        seen_keys.add(call.idempotency_key)
        allowed, reason = collection_eligibility(row)
        if not allowed:
            issues.append({"prompt_id": call.prompt_id, "reason": reason})
    if issues:
        preview = "; ".join(f"{item['prompt_id']}: {item['reason']}" for item in issues[:8])
        raise StudyCollectorError(
            "all-call static preflight failed before any reservation/provider call: "
            f"{len(issues)} issue(s); {preview}"
        )
    return {
        "status": "passed",
        "planned_call_count": len(calls),
        "idempotency_key_count": len(seen_keys),
        "provider_calls": 0,
        "reservation_count": 0,
        "executor_manifest": str(executor_manifest) if manifest is not None else None,
        "executor_manifest_hash": manifest.get("manifest_hash") if manifest else None,
        "grader_runtime_selection": {
            role: {
                "requested_executable": record.get("requested_executable"),
                "resolved_executable": record.get("resolved_executable"),
                "python_identity": record.get("python_identity"),
                "environment_hash": record.get("environment_hash"),
                "reproducible_contract_hash": record.get("reproducible_contract_hash"),
            }
            for role, record in (manifest.get("runtimes") or {}).items()
        }
        if manifest
        else {},
        "sandbox_policy_version": ((manifest or {}).get("sandbox") or {}).get("policy_version"),
        "public_anchor_overlap_manifest": str(overlap_manifest),
        "anchor_boundary_hash": boundary.manifest_hash,
        "public_anchor_outcomes_used_for_overlap": False,
    }


def _check_cost_gates(plan: StudyPlan) -> None:
    for name, scenario in plan.cost_scenarios.items():
        if scenario.get("status") != "priced":
            raise StudyCollectorError(f"{name} is not fully priced")
        if scenario.get("within_hard_cap") is not True:
            raise StudyCollectorError(f"{name} exceeds the $25 hard cap")
        if scenario.get("untouched_test_reserve_preserved") is not True:
            raise StudyCollectorError(f"{name} would consume the untouched-test reserve")


def _default_execution_plan_path(phase: str) -> Path:
    slug = phase.replace("_", "-")
    return ROOT / "reports/router-v2" / f"router-v2-{slug}-executable-plan.json"


def _metadata_for_plan(
    prompts: tuple[Any, ...],
    plan: StudyPlan,
    *,
    expected_response_sha256: str | None = None,
) -> dict[str, Any]:
    calls = plan.planned_calls or plan.calls
    model_ids = tuple(dict.fromkeys(call.model for call in calls))
    has_tools = any(adapter_for_prompt(row).adapter_id != "plain-text" for row in prompts)
    metadata = fetch_openrouter_metadata(
        model_ids,
        tool_model_ids=model_ids if has_tools else (),
    )
    compare_metadata_to_plan(
        metadata,
        plan.price_snapshot,
        expected_response_sha256=expected_response_sha256,
    )
    return metadata


def _metadata_matches_frozen_snapshot(
    current: Mapping[str, Any], frozen: Mapping[str, Any]
) -> bool:
    """Compare executable metadata while ignoring audit-only fetch provenance."""

    current_value = dict(current)
    frozen_value = dict(frozen)
    for key in ("fetched_at", "catalog_response_sha256"):
        current_value.pop(key, None)
        frozen_value.pop(key, None)
    return current_value == frozen_value


def _verify_execution_now(
    *,
    execution_path: Path,
    parent_path: Path,
    phase: str,
    corpus_path: Path,
    overlap_path: Path,
    manifest_path: Path,
) -> Any:
    """Re-read every hash-bound input immediately before provider creation."""

    executable = load_execution_plan(execution_path)
    parent, parent_hash, phased_hash = _load_phase(parent_path, phase)
    prompts = tuple(load_jsonl(corpus_path))
    static = _static_preflight(
        prompts,
        parent,
        overlap_manifest=overlap_path,
        executor_manifest=manifest_path,
    )
    manifest_hash = str(static.get("executor_manifest_hash") or "")
    assert_matches_locked_plan(
        executable,
        parent,
        parent_phased_plan_hash=phased_hash,
        corpus_hash=corpus_checksum(prompts),
        executor_manifest_hash=manifest_hash,
        anchor_boundary_hash=str(static["anchor_boundary_hash"]),
    )
    metadata = _metadata_for_plan(
        prompts,
        parent,
        expected_response_sha256=executable.metadata_response_sha256,
    )
    if not _metadata_matches_frozen_snapshot(metadata, executable.metadata_snapshot):
        raise ExecutionPlanError("public model metadata changed since executable-plan review")
    if parent_hash != executable.parent_plan_hash:
        raise ExecutionPlanError("parent phase hash changed before provider construction")
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=STUDY_PHASES, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--reviewed-plan-hash")
    parser.add_argument(
        "--anchor-overlap-manifest",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-public-anchor-overlap-manifest.json",
    )
    parser.add_argument(
        "--executor-manifest",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-executor-environment-manifest.json",
    )
    parser.add_argument(
        "--metadata-report",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-openrouter-model-metadata-snapshot.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument(
        "--retry-error-prompt-id",
        action="append",
        default=[],
        help=(
            "Retry one explicitly reviewed prompt whose prior provider error has "
            "a settled reservation; repeat for additional prompt IDs."
        ),
    )
    parser.add_argument(
        "--only-model",
        action="append",
        default=[],
        help=(
            "Execute only calls for this model from the reviewed plan; repeat for "
            "additional models. A selected-model run never completes the phase."
        ),
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        prompts = tuple(load_jsonl(args.corpus))
        audit = validate_corpus(prompts)
        parent, parent_phase_hash, parent_phased_hash = _load_phase(args.plan, args.phase)
        static = _static_preflight(
            prompts,
            parent,
            overlap_manifest=args.anchor_overlap_manifest,
            executor_manifest=args.executor_manifest,
        )
        _check_cost_gates(parent)
        ledger = StudyBudgetLedger(
            args.budget_ledger,
            StudyBudgetConfig(hard_cap_usd=25.0, test_reserve_usd=10.0),
        )
        execution_path = args.execution_plan or _default_execution_plan_path(args.phase)
        executable = None
        metadata = None
        if args.execute:
            if args.execution_plan is None:
                raise ExecutionPlanError("execution requires --execution-plan; locked plans are not executable")
            executable = load_execution_plan(args.execution_plan)
            assert_matches_locked_plan(
                executable,
                parent,
                parent_phased_plan_hash=parent_phased_hash,
                corpus_hash=corpus_checksum(prompts),
                executor_manifest_hash=str(static.get("executor_manifest_hash") or ""),
                anchor_boundary_hash=str(static["anchor_boundary_hash"]),
            )
            metadata = _metadata_for_plan(
                prompts,
                parent,
                expected_response_sha256=executable.metadata_response_sha256,
            )
            if not _metadata_matches_frozen_snapshot(metadata, executable.metadata_snapshot):
                raise ExecutionPlanError("public model metadata changed since executable-plan review")
        else:
            metadata = _metadata_for_plan(prompts, parent)
            write_metadata_snapshot(metadata, args.metadata_report)
            if execution_path.exists():
                executable = load_execution_plan(execution_path)
                _metadata_for_plan(
                    prompts,
                    parent,
                    expected_response_sha256=executable.metadata_response_sha256,
                )
                if not _metadata_matches_frozen_snapshot(metadata, executable.metadata_snapshot):
                    raise ExecutionPlanError(
                        "public model metadata changed; regenerate and re-review the executable plan"
                    )
            else:
                executable = materialize_execution_plan(
                    parent,
                    parent_phased_plan_hash=parent_phased_hash,
                    executor_manifest_hash=str(static.get("executor_manifest_hash") or ""),
                    anchor_boundary_hash=str(static["anchor_boundary_hash"]),
                    metadata_snapshot=metadata,
                    metadata_response_sha256=str(metadata["response_sha256"]),
                    metadata_fetched_at=str(metadata["fetched_at"]),
                )
                executable_bytes = write_execution_plan(executable, execution_path)
                executable = load_execution_plan(execution_path)
                if executable_bytes != execution_path.read_bytes():
                    raise ExecutionPlanError("executable plan changed during provider-free preflight")
            assert_matches_locked_plan(
                executable,
                parent,
                parent_phased_plan_hash=parent_phased_hash,
                corpus_hash=corpus_checksum(prompts),
                executor_manifest_hash=str(static.get("executor_manifest_hash") or ""),
                anchor_boundary_hash=str(static["anchor_boundary_hash"]),
            )

        if args.reviewed_plan_hash is not None:
            assert_reviewed_hash(executable, args.reviewed_plan_hash)

        report: dict[str, Any] = {
            "schema_version": 2,
            "mode": "execute" if args.execute else "preflight",
            "phase": args.phase,
            "provider_calls": 0,
            "provider_api_invocations": 0,
            "task_model_runs": 0,
            "trajectory_turns": 0,
            "completed_graded_outcomes": 0,
            "reserved_cost_usd": 0.0,
            "outstanding_reserved_cost_usd": 0.0,
            "realized_provider_cost_usd": 0.0,
            "provider_spend_usd": 0.0,
            "test_outcomes_opened": False,
            "test_outcomes_status": "test prompts are frozen and inspectable; candidate-model test outcomes remain unopened.",
            "router_v2_enabled": False,
            "corpus_checksum": corpus_checksum(prompts),
            "corpus_audit": audit.to_dict(),
            "plan_path": str(args.plan),
            "locked_phase_plan_hash": parent_phase_hash,
            "parent_plan_hash": parent_phase_hash,
            "parent_phased_plan_hash": parent_phased_hash,
            "execution_plan_path": str(execution_path),
            "execution_plan_hash": executable.execution_plan_hash,
            "execution_plan_file_sha256": file_sha256(execution_path),
            "locked_and_execution_hashes_intentionally_differ": (
                parent_phase_hash != executable.execution_plan_hash
            ),
            "planned_call_count": executable.call_count,
            "materialized_call_count_before_execution": executable.call_count,
            "static_preflight": static,
            "metadata_snapshot": metadata,
            "metadata_response_sha256": executable.metadata_response_sha256,
            "metadata_fetched_at": executable.metadata_fetched_at,
            "runtime_provenance": {
                "sys_version": sys.version,
                "sys_executable": sys.executable,
                "resolved_executable": os.path.realpath(sys.executable),
                "platform": platform.platform(),
                "sandbox_policy_version": static.get("sandbox_policy_version"),
                "exact_test_command": shlex.join(sys.argv),
                "grader_runtime_selection": static.get("grader_runtime_selection", {}),
            },
            "budget_ledger": {
                "path": str(args.budget_ledger),
                "status": ledger.status(),
                "accounting": ledger.accounting(),
            },
        }
        if not args.execute:
            if args.allow_provider_calls or args.allow_paid_calls or args.yes:
                raise StudyCollectorError("approval flags are only valid with --execute")
            report["next_step"] = "review this immutable executable phase plan; no provider call was made"
        else:
            if not args.allow_provider_calls or not args.yes:
                raise StudyCollectorError("execution requires --allow-provider-calls and --yes")
            if not args.reviewed_plan_hash:
                raise StudyCollectorError("execution requires --reviewed-plan-hash")
            verified = _verify_execution_now(
                execution_path=args.execution_plan,
                parent_path=args.plan,
                phase=args.phase,
                corpus_path=args.corpus,
                overlap_path=args.anchor_overlap_manifest,
                manifest_path=args.executor_manifest,
            )
            collector = StudyCollector(
                prompts=prompts,
                plan=verified.to_study_plan(),
                ledger=ledger,
                state_path=args.state,
                adapter_context={"enforce_price_snapshot": True},
                pre_provider_verifier=lambda: _verify_execution_now(
                    execution_path=args.execution_plan,
                    parent_path=args.plan,
                    phase=args.phase,
                    corpus_path=args.corpus,
                    overlap_path=args.anchor_overlap_manifest,
                    manifest_path=args.executor_manifest,
                ),
            )
            execution = collector.execute(
                reviewed_plan_hash=verified.execution_plan_hash,
                allow_provider_calls=args.allow_provider_calls,
                allow_paid_calls=args.allow_paid_calls,
                yes=args.yes,
                retry_errors=bool(args.retry_error_prompt_id),
                retry_error_prompt_ids=frozenset(args.retry_error_prompt_id),
                selected_models=frozenset(args.only_model),
            )
            report["execution"] = execution
            report["task_model_runs"] = execution.get("task_model_runs", 0)
            report["provider_api_invocations"] = execution.get("provider_api_invocations", 0)
            report["trajectory_turns"] = execution.get("trajectory_turns", 0)
            report["completed_graded_outcomes"] = execution.get("completed_graded_outcomes", 0)
            report["reserved_cost_usd"] = execution.get("reserved_cost_usd", 0.0)
            report["outstanding_reserved_cost_usd"] = execution.get(
                "outstanding_reserved_cost_usd", 0.0
            )
            report["realized_provider_cost_usd"] = execution.get(
                "realized_provider_cost_usd", execution.get("actual_cost_usd", 0.0)
            )
            report["provider_calls"] = report["provider_api_invocations"]
            report["provider_spend_usd"] = report["realized_provider_cost_usd"]
            report["test_outcomes_opened"] = bool(
                args.phase == PHASE_D and report["provider_api_invocations"] > 0
            )
            if report["test_outcomes_opened"]:
                report["test_outcomes_status"] = "candidate-model test outcomes were opened by a Phase D provider call."
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        StudyCorpusError,
        StudyCollectorError,
        ExecutorManifestError,
        ExecutionPlanError,
        OpenRouterMetadataError,
        ProviderError,
        CircuitOpenError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"router-v2 phase stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
