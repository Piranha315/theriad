#!/usr/bin/env python3
"""Emit a provider-free Router V2 audit, including replacement gaps.

This command intentionally uses ``audit_corpus`` rather than promotion
validation so a blocked corpus still gets a complete evidence report. It never
loads model outcomes and never calls a provider.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import (  # noqa: E402
    CORE_MODELS,
    FORMAT_TOKEN_BUDGETS,
    GROK_MODEL,
    GROK_TRAIN_COUNT,
    STUDY_SPLIT_COUNTS,
    audit_corpus,
    corpus_checksum,
    load_jsonl,
)
from theriad.routing.study_adapters import (  # noqa: E402
    adapter_for_prompt,
    collection_eligibility,
    run_bfcl_smoke_fixtures,
    run_tau_environment_smoke,
)
from theriad.routing.study_anchor_boundary import StudyAnchorExclusionBoundary  # noqa: E402
from theriad.routing.study_budget import StudyBudgetLedger  # noqa: E402
from theriad.routing.study_graders import validate_pairwise_contract  # noqa: E402


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _executor_status(official_smoke: dict[str, Any] | None = None) -> dict[str, Any]:
    official_status = str((official_smoke or {}).get("status") or "missing")
    bfcl_status = str(((official_smoke or {}).get("bfcl") or {}).get("status") or "missing")
    tau_status = str(((official_smoke or {}).get("tau2") or {}).get("status") or "missing")
    return {
        "bfcl": {
            "command": shutil.which("bfcl"),
            "module": bool(importlib.util.find_spec("bfcl")),
            "official_checker_executable": bfcl_status == "passed",
            "pinned_smoke_status": bfcl_status,
        },
        "tau_bench": {
            "commands": {
                name: shutil.which(name) for name in ("tau2", "tau-bench")
            },
            "modules": {
                name: bool(importlib.util.find_spec(name))
                for name in ("tau2", "tau_bench")
            },
            "official_environment_executable": tau_status == "passed",
            "pinned_smoke_status": tau_status,
        },
        "pinned_smoke_report_status": official_status,
    }


def _pairwise_plan(rows: tuple[Any, ...]) -> dict[str, Any]:
    pairwise = [row for row in rows if row.label_type == "pairwise"]
    contracts = []
    errors = []
    for row in pairwise:
        try:
            validate_pairwise_contract(row.grader)
            contracts.append(row.prompt_id)
        except ValueError as exc:
            errors.append({"prompt_id": row.prompt_id, "reason": str(exc)})
    statuses = Counter(str(row.grader.get("judging_cost_status") or "missing") for row in pairwise)
    report = {
        "row_count": len(pairwise),
        "contract_valid_count": len(contracts),
        "contract_errors": errors,
        "judge_identity_versions": sorted(
            {f"{row.grader.get('judge_id')}@{row.grader.get('judge_version')}" for row in pairwise}
        ),
        "rubric_versions": sorted({str(row.grader.get("criteria_version")) for row in pairwise}),
        "randomization": sorted({str(row.grader.get("position_randomization")) for row in pairwise}),
        "sampling_settings": sorted(
            {
                json.dumps(row.grader.get("sampling_settings"), sort_keys=True)
                for row in pairwise
            }
        ),
        "tie_policies": sorted({str(row.grader.get("tie_policy")) for row in pairwise}),
        "abstention_policies": sorted(
            {str(row.grader.get("abstention_policy")) for row in pairwise}
        ),
        "disagreement_review_policies": sorted(
            {str(row.grader.get("disagreement_review_policy")) for row in pairwise}
        ),
        "judging_cost_status_counts": dict(statuses),
        "expected_judging_cost_usd": None,
        "worst_case_judging_cost_usd": None,
        "usable_outcome_policy": "postponed pairwise rows are excluded, never counted as zero-cost completed grades",
    }
    if not pairwise:
        report.update(
            {
                "policy": "removed_no_judging_calls",
                "judging_cost_status_counts": {},
                "expected_judging_cost_usd": 0.0,
                "worst_case_judging_cost_usd": 0.0,
                "usable_outcome_policy": "no pairwise rows retained; zero judge calls are planned",
            }
        )
    return report


def _phase_schedule() -> list[dict[str, Any]]:
    return [
        {
            "phase": "A_core_train",
            "models": list(CORE_MODELS),
            "split": "train",
            "call_count": 3 * STUDY_SPLIT_COUNTS["train"],
            "status": "blocked_until_corpus_promotion",
            "selection_ids_status": "not_applicable",
        },
        {
            "phase": "B_grok_train",
            "models": [GROK_MODEL],
            "split": "train",
            "call_count": GROK_TRAIN_COUNT,
            "materialized_call_count_before_phase_A": 0,
            "status": "locked_ids_unresolved_until_phase_A_outcomes",
            "selection_ids_status": "deferred_until_complete_core_outcomes",
        },
        {
            "phase": "C_validation",
            "models": [*CORE_MODELS, GROK_MODEL],
            "split": "validation",
            "call_count": 4 * STUDY_SPLIT_COUNTS["validation"],
            "status": "locked_until_A_and_B_complete",
        },
        {
            "phase": "D_test",
            "models": [*CORE_MODELS, GROK_MODEL],
            "split": "test",
            "call_count": 4 * STUDY_SPLIT_COUNTS["test"],
            "status": "locked_until_final_freeze_manifest",
            "test_outcomes_opened": False,
        },
    ]


def _load_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _plan_schedule(plan: dict[str, Any]) -> list[dict[str, Any]]:
    phases = plan.get("phases")
    if not isinstance(phases, dict):
        return _phase_schedule()
    schedule = []
    for name in ("A_core_train", "B_grok_train", "C_validation", "D_test"):
        phase = phases.get(name) or {}
        schedule.append(
            {
                "phase": name,
                "models": sorted((phase.get("price_snapshot") or {}).keys()),
                "split": {
                    "A_core_train": "train",
                    "B_grok_train": "train",
                    "C_validation": "validation",
                    "D_test": "test",
                }[name],
                "call_count": int(phase.get("call_count") or 0),
                "potential_call_count": int(phase.get("potential_call_count") or 0),
                "status": str(phase.get("status") or "unknown"),
                "plan_hash": str(phase.get("plan_hash") or ""),
                "selection_ids_status": str(
                    (phase.get("grok_selection") or {}).get("selected_ids_status")
                    or "not_applicable"
                ),
                "test_outcomes_opened": name == "D_test" and False,
            }
        )
    return schedule


def _aggregate_plan_costs(plan: dict[str, Any]) -> dict[str, Any]:
    phases = plan.get("phases") if isinstance(plan.get("phases"), dict) else {}
    result: dict[str, Any] = {
        "status": "priced" if phases else "missing_plan",
        "hard_cap_usd": 25.0,
        "untouched_test_reserve_usd": 10.0,
        "pairwise_judging_cost_expected_usd": 0.0,
        "pairwise_judging_cost_worst_case_usd": 0.0,
    }
    for name in ("free_nemotron", "paid_nemotron"):
        entries = [
            (phase.get("cost_scenarios") or {}).get(name)
            for phase in phases.values()
            if isinstance(phase, dict)
        ]
        entries = [entry for entry in entries if isinstance(entry, dict)]
        expected = sum(float(entry.get("expected_total_cost_usd") or 0.0) for entry in entries)
        worst = sum(float(entry.get("worst_case_total_cost_usd") or 0.0) for entry in entries)
        non_test = sum(float(entry.get("worst_case_non_test_cost_usd") or 0.0) for entry in entries)
        test = sum(float(entry.get("worst_case_test_cost_usd") or 0.0) for entry in entries)
        result[name] = {
            "status": "priced" if entries and all(entry.get("status") == "priced" for entry in entries) else "missing_pinned_quote",
            "expected_model_and_grading_cost_usd": expected,
            "worst_case_model_and_grading_cost_usd": worst,
            "expected_total_cost_usd": expected,
            "worst_case_total_cost_usd": worst,
            "worst_case_non_test_cost_usd": non_test,
            "worst_case_test_cost_usd": test,
            "within_hard_cap": worst <= 25.0 + 1e-12 if entries else None,
            "untouched_test_reserve_preserved": non_test <= 15.0 + 1e-12 if entries else None,
            "grading_cost_usd": 0.0,
            "grading_cost_status": "local_deterministic_only; zero_pairwise_rows",
            "by_phase": {
                str(phase.get("phase")): {
                    "expected_total_cost_usd": float(
                        ((phase.get("cost_scenarios") or {}).get(name) or {}).get(
                            "expected_total_cost_usd", 0.0
                        )
                    ),
                    "worst_case_total_cost_usd": float(
                        ((phase.get("cost_scenarios") or {}).get(name) or {}).get(
                            "worst_case_total_cost_usd", 0.0
                        )
                    ),
                }
                for phase in phases.values()
                if isinstance(phase, dict)
            },
        }
    return result


def _budget_guard(plan_costs: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    scenarios: dict[str, dict[str, Any]] = {}
    for name in ("free_nemotron", "paid_nemotron"):
        entry = plan_costs.get(name)
        if not isinstance(entry, dict):
            blockers.append(f"{name}: missing aggregate cost quote")
            continue
        within_hard_cap = entry.get("within_hard_cap") is True
        reserve_preserved = entry.get("untouched_test_reserve_preserved") is True
        if not within_hard_cap:
            blockers.append(
                f"{name}: worst-case total exceeds the $25 hard cap"
            )
        if not reserve_preserved:
            blockers.append(
                f"{name}: worst-case non-test cost exceeds the $15 spendable envelope"
            )
        scenarios[name] = {
            "worst_case_total_cost_usd": entry.get("worst_case_total_cost_usd"),
            "worst_case_non_test_cost_usd": entry.get("worst_case_non_test_cost_usd"),
            "worst_case_test_cost_usd": entry.get("worst_case_test_cost_usd"),
            "within_hard_cap": within_hard_cap,
            "untouched_test_reserve_preserved": reserve_preserved,
        }
    return {
        "status": "passed" if not blockers else "blocked",
        "hard_cap_usd": 25.0,
        "untouched_test_reserve_usd": 10.0,
        "spendable_non_test_envelope_usd": 15.0,
        "scenarios": scenarios,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--bfcl-cache-dir", type=Path)
    parser.add_argument("--tau3-cache-dir", type=Path)
    parser.add_argument(
        "--anchor-overlap-manifest",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-public-anchor-overlap-manifest.json",
    )
    parser.add_argument(
        "--official-executor-smoke",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-official-executor-smoke.json",
    )
    parser.add_argument(
        "--source-acquisition-report",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-source-acquisition.json",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.corpus)
    official_smoke = _load_report(args.official_executor_smoke)
    source_acquisition = _load_report(args.source_acquisition_report)
    existing_plan = _load_report(args.plan_output)
    audit = audit_corpus(rows)
    anchor_boundary: dict[str, Any]
    try:
        boundary = StudyAnchorExclusionBoundary.load(args.anchor_overlap_manifest)
        if boundary.study_corpus_checksum != corpus_checksum(rows):
            raise ValueError("study-anchor overlap manifest belongs to a different corpus")
        anchor_boundary = {
            "status": "passed",
            **boundary.to_dict(),
            "path": str(args.anchor_overlap_manifest),
        }
    except (OSError, TypeError, ValueError) as exc:
        anchor_boundary = {
            "status": "blocked",
            "path": str(args.anchor_overlap_manifest),
            "error": f"{type(exc).__name__}: {exc}",
        }
    eligibility = Counter()
    adapter_counts = Counter()
    code_lane_counts = Counter()
    executable_code_counts = Counter()
    code_tool_state_counts = Counter()
    family_coverage: dict[str, dict[str, int]] = {}
    for row in rows:
        adapter_counts[adapter_for_prompt(row).adapter_id] += 1
        allowed, reason = collection_eligibility(row)
        eligibility["eligible" if allowed else reason] += 1
        family = row.task_format
        family_row = family_coverage.setdefault(
            family,
            {"rows": 0, "adapter_eligible": 0, "usable_outcome_rows": 0, "pending_grading_rows": 0},
        )
        family_row["rows"] += 1
        family_row["adapter_eligible"] += int(allowed)
        family_row["pending_grading_rows"] += int(
            row.label_type == "pairwise"
            and row.grader.get("judging_cost_status") == "postponed_unpriced"
        )
        family_row["usable_outcome_rows"] += int(
            allowed
            and not (
                row.label_type == "pairwise"
                and row.grader.get("judging_cost_status") == "postponed_unpriced"
            )
        )
        if row.lane == "code":
            code_lane_counts[row.split] += 1
            if row.label_type == "tool_state":
                code_tool_state_counts[row.split] += 1
        if (
            row.task_format == "code_execution"
            and row.label_type == "deterministic"
            and str(row.grader.get("kind") or "") == "sandboxed_code_tests"
        ):
            executable_code_counts[row.split] += 1
    executors = _executor_status(official_smoke)
    bfcl_smoke = {
        "status": str(((official_smoke.get("bfcl") or {}).get("status")) or "missing"),
        "official_checker_executable": str(
            ((official_smoke.get("bfcl") or {}).get("status")) or "missing"
        )
        == "passed",
        "official_evidence_path": str(args.official_executor_smoke),
        "official_checker": (official_smoke.get("bfcl") or {}).get("official_checker", {}),
        "adapter_fixture_compatibility": run_bfcl_smoke_fixtures(),
        "provider_calls": 0,
    }
    tau_smoke = {
        "status": str(((official_smoke.get("tau2") or {}).get("status")) or "missing"),
        "official_environment_executable": str(
            ((official_smoke.get("tau2") or {}).get("status")) or "missing"
        )
        == "passed",
        "official_evidence_path": str(args.official_executor_smoke),
        "official_environment": (official_smoke.get("tau2") or {}).get(
            "official_environment", {}
        ),
        "adapter_fixture_compatibility": run_tau_environment_smoke(),
        "provider_calls": 0,
    }
    pairwise = _pairwise_plan(rows)
    ledger = StudyBudgetLedger(args.budget_ledger)
    bfcl_cached = _count_jsonl(args.bfcl_cache_dir / "BFCL_v4_irrelevance.json") if args.bfcl_cache_dir else None
    tau_cached = 0
    if args.tau3_cache_dir:
        legacy_files = [
            args.tau3_cache_dir / f"data_tau2_domains_{domain}_tasks.json"
            for domain in ("airline", "mock", "retail", "telecom")
        ]
        tau_cached = sum(_count_jsonl(path) for path in legacy_files)
        if tau_cached == 0:
            tau_cached = sum(
                len(json.loads((args.tau3_cache_dir / "data" / "tau2" / "domains" / domain / "tasks.json").read_text(encoding="utf-8")))
                for domain in ("airline", "mock", "retail", "telecom", "banking_knowledge")
                if (args.tau3_cache_dir / "data" / "tau2" / "domains" / domain / "tasks.json").exists()
            )
    plan_schedule = _plan_schedule(existing_plan)
    plan_costs = _aggregate_plan_costs(existing_plan)
    budget_guard = _budget_guard(plan_costs)
    audit_ready = (
        audit.valid
        and budget_guard["status"] == "passed"
        and anchor_boundary["status"] == "passed"
    )
    report = {
        "schema_version": 1,
        "status": "ready_for_dry_run_plan" if audit_ready else "blocked",
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "test_outcomes_opened": False,
        "production_artifact_trained": False,
        "router_v2_enabled": False,
        "corpus": {
            "path": str(args.corpus),
            "checksum": corpus_checksum(rows),
            "audit": audit.to_dict(),
            "lane_distribution": audit.lane_counts,
            "source_distribution": audit.source_counts,
            "task_format_distribution": audit.task_format_counts,
            "label_distribution": audit.label_counts,
            "stratum_distribution": audit.stratum_counts,
            "public_anchor_overlap_boundary": anchor_boundary,
        },
        "adapter_eligibility": {
            "counts": dict(eligibility),
            "adapter_counts": dict(adapter_counts),
            "coverage_by_task_family": family_coverage,
            "executors": executors,
            "bfcl_cached_inventory_rows": bfcl_cached,
            "tau_cached_inventory_rows": tau_cached,
        },
        "bfcl_official_checker": bfcl_smoke,
        "tau_bench_environment": tau_smoke,
        "genuine_code_lane": {
            "counts_by_split": dict(code_lane_counts),
            "required_by_split": {"train": 240, "validation": 80, "test": 80},
            "shortfall_by_split": {
                split: required - code_lane_counts.get(split, 0)
                for split, required in {"train": 240, "validation": 80, "test": 80}.items()
            },
            "deterministic_executable_code_counts_by_split": dict(executable_code_counts),
            "code_tool_state_counts_by_split": dict(code_tool_state_counts),
            "licensed_pinned_executable_sources": sorted(
                {
                    row.source
                    for row in rows
                    if row.task_format == "code_execution"
                    and row.label_type == "deterministic"
                    and str(row.grader.get("kind") or "") == "sandboxed_code_tests"
                }
            ),
            "bfcl_is_not_code": True,
            "definition": (
                "lane=code coverage includes the separately identified BFCL tool-use capability "
                "slice; only code_execution rows count as deterministic executable code"
            ),
        },
        "pairwise_judging": pairwise,
        "format_specific_token_budgets": FORMAT_TOKEN_BUDGETS,
        "phased_call_schedule": plan_schedule,
        "plan": {
            "path": str(args.plan_output) if args.plan_output else None,
            "plan_hash": existing_plan.get("plan_hash"),
            "corpus_hash": existing_plan.get("corpus_hash"),
            "provider_calls_materialized": existing_plan.get("provider_calls_materialized", 0),
            "potential_call_count": existing_plan.get("potential_call_count", 0),
            "idempotency_key_template": "{phase}:{prompt_id}:{model_id}",
            "test_unlock": False,
        },
        "costs": {
            **plan_costs,
            "output_token_and_grading_cost": {
                "format_budgets_frozen": True,
                "pairwise_judging_cost_expected_usd": 0.0,
                "pairwise_judging_cost_worst_case_usd": 0.0,
            },
        },
        "budget_guard": budget_guard,
        "budget_ledger": {"path": str(args.budget_ledger), "status": ledger.status()},
        "source_acquisition": {
            "path": str(args.source_acquisition_report),
            "status": "passed" if source_acquisition.get("audit_before_write", {}).get("valid") else "missing_or_invalid",
            "corpus_checksum": source_acquisition.get("checksum"),
            "output_sha256": source_acquisition.get("output_sha256"),
        },
        "unresolved_data_gaps": [
            "No accepted code source with explicit compatible license supplied repository-level repair tasks; SWE-bench remains excluded pending per-item license and sandbox review.",
            "No accepted code source with explicit compatible license supplied a separate debugging/self-repair slice; LiveCodeBench remains excluded pending dataset-license clarification.",
            "Authentic model-disagreement remains a post-collection evaluation slice; no candidate-model outcomes are present in this audit.",
        ],
    }
    if budget_guard["blockers"]:
        report["unresolved_data_gaps"].append(
            "The aggregate phased worst-case envelope exceeds the hard $25 cap and/or the $10 untouched-test reserve; no paid phase is unlocked."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if audit_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
