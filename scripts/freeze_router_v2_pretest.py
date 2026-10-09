#!/usr/bin/env python3
"""Write and verify the locked pre-test Router V2 freeze manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.accounting.pricing import PricingTable  # noqa: E402
from theriad.routing.sandbox import (  # noqa: E402
    SANDBOX_EXECUTOR_ID,
    SANDBOX_POLICY_VERSION,
)
from theriad.routing.study import CORE_MODELS, GROK_MODEL, load_jsonl  # noqa: E402
from theriad.routing.study_adapters import (  # noqa: E402
    BFCL_ADAPTER_VERSION,
    CODE_ADAPTER_VERSION,
    TAU_ADAPTER_VERSION,
)
from theriad.routing.study_freeze import (  # noqa: E402
    build_pretest_freeze_manifest,
    validate_pretest_freeze_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-phased-call-plan.json",
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
    parser.add_argument(
        "--executor-manifest",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-executor-environment-manifest.json",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.corpus)
    state = {}
    if args.state and args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
    plan = (
        json.loads(args.plan.read_text(encoding="utf-8"))
        if args.plan.exists()
        else {}
    )
    official_smoke = (
        json.loads(args.official_executor_smoke.read_text(encoding="utf-8"))
        if args.official_executor_smoke.exists()
        else {}
    )
    source_report = (
        json.loads(args.source_acquisition_report.read_text(encoding="utf-8"))
        if args.source_acquisition_report.exists()
        else {}
    )
    executor_report = (
        json.loads(args.executor_manifest.read_text(encoding="utf-8"))
        if args.executor_manifest.exists()
        else {}
    )
    plan_hashes = dict(state.get("plan_hashes") or {})
    if plan.get("plan_hash"):
        plan_hashes.setdefault("phased_plan", str(plan["plan_hash"]))
    for phase_name, phase in (plan.get("phases") or {}).items():
        if isinstance(phase, dict) and phase.get("plan_hash"):
            plan_hashes.setdefault(str(phase_name), str(phase["plan_hash"]))
    grok_algorithm = dict(plan.get("grok_algorithm") or {})
    selected_ids = list(state.get("grok_selected_ids") or grok_algorithm.get("prompt_ids") or [])
    bfcl_smoke = dict(official_smoke.get("bfcl") or {})
    tau_smoke = dict(official_smoke.get("tau2") or {})
    pricing = PricingTable(use_live=False)
    price_snapshot = {}
    for model in (*CORE_MODELS, GROK_MODEL):
        lookup = pricing.lookup(model)
        if lookup is not None:
            price_snapshot[model] = {
                "input_per_1m": float(lookup.price.input_per_1m),
                "output_per_1m": float(lookup.price.output_per_1m),
                "source": lookup.source,
                "pricing_model_slug": lookup.model_slug,
            }
    paid_lookup = pricing.lookup("nvidia/nemotron-3-super-120b-a12b")
    if paid_lookup is not None:
        price_snapshot["nvidia/nemotron-3-super-120b-a12b:paid_scenario"] = {
            "input_per_1m": float(paid_lookup.price.input_per_1m),
            "output_per_1m": float(paid_lookup.price.output_per_1m),
            "source": paid_lookup.source,
            "pricing_model_slug": paid_lookup.model_slug,
        }
    manifest = build_pretest_freeze_manifest(
        rows,
        phase_status=state.get("phase_status") or {},
        plan_hashes=plan_hashes,
        price_snapshot=price_snapshot,
        task_adapters={
            "plain-text": "router-v2-plain-text-v1",
            "bfcl": {
                "adapter_version": BFCL_ADAPTER_VERSION,
                "official_checker_id": "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker",
                "official_checker_revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
                "official_checker_status": bfcl_smoke.get("status", "missing"),
                "structured_tool_calls": ["name", "arguments", "ordering", "parallel_group"],
                "candidate_without_tool_calling": "ineligible",
            },
            "tau-bench": {
                "adapter_version": TAU_ADAPTER_VERSION,
                "official_environment_id": "tau2.evaluator.evaluator.evaluate_simulation",
                "official_environment_revision": "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
                "official_environment_status": tau_smoke.get("status", "missing"),
                "interactive_state_loop": True,
                "persisted_environment_state": True,
            },
            "executable-code": {
                "adapter_version": CODE_ADAPTER_VERSION,
                "sandbox_executor": SANDBOX_EXECUTOR_ID,
                "sandbox_policy_version": SANDBOX_POLICY_VERSION,
                "network_disabled": True,
                "base_filesystem_read_only": True,
                "task_directory_only_writable": True,
                "host_code_execution": False,
                "posix_shared_memory_for_pinned_numeric_runtime": True,
            },
            "executor_environment_manifest": {
                "path": str(args.executor_manifest),
                "status": executor_report.get("status", "missing"),
                "manifest_hash": executor_report.get("manifest_hash"),
                "runtimes": executor_report.get("runtimes", {}),
                "sandbox": executor_report.get("sandbox", {}),
            },
        },
        sampling_settings={
            "candidate_models": {"temperature": 0.0, "top_p": 1.0, "attempts": 1},
            "pairwise": {"row_count": 0, "judge_calls": 0, "status": "removed"},
        },
        grader_versions={
            "deterministic": "router-v2-grader-2026-08-24-v1",
            "bfcl_official_checker": "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
            "tau_bench_official_evaluator": "tau2.evaluator.evaluator.evaluate_simulation@a2c024725189473d2d7cea3a5cfdbcc67478e41f",
            "code_sandbox": f"{SANDBOX_EXECUTOR_ID}@{SANDBOX_POLICY_VERSION}",
            "pairwise": "removed_zero_rows_zero_judge_calls",
        },
        feature_set=[
            "prompt_length",
            "task_format",
            "lane",
            "source_distance",
            "uncertainty",
            "cost",
            "latency",
        ],
        router_families=["surface_baseline", "linear_router", "contrastive_router"],
        encoders={
            "baseline": {"name": "tfidf", "revision": "router-v2-tfidf-v1"},
            "semantic": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "locked_not_acquired_before_training",
            },
        },
        calibration_method={"name": "validation_only", "status": "pending_phase_C"},
        lower_confidence_bound_method={
            "name": "validation_bootstrap_lcb",
            "revision": "router-v2-lcb-v1",
        },
        knee_selection_algorithm={
            "name": "automatic_pareto_knee",
            "revision": "router-v2-knee-v1",
            "source": "theriad.routing.v2_training.automatic_knee",
        },
        grok_selection={
            **grok_algorithm,
            "algorithm_version": str(
                grok_algorithm.get("algorithm_version") or "router-v2-grok-informative-v2"
            ),
            "prompt_ids": selected_ids,
            "selected_ids": selected_ids,
            "selected_ids_status": "frozen" if selected_ids else "unresolved_until_phase_A_outcomes",
            "selection_evidence_status": "not_observed_before_phase_A" if not selected_ids else "state_supplied",
        },
        source_revisions=sorted({row.source_revision for row in rows}),
        provider_identities={
            "models": [*CORE_MODELS, GROK_MODEL],
            "provider": "openrouter",
            "model_provider_identities": {
                model: f"openrouter:{model}" for model in (*CORE_MODELS, GROK_MODEL)
            },
            "identity_status": "pinned_model_ids; provider call authorization absent",
            "source_acquisition_checksum": source_report.get("checksum"),
        },
    )
    validate_pretest_freeze_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
