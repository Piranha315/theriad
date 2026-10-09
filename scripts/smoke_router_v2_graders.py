#!/usr/bin/env python3
"""Run local, provider-free smoke fixtures for every Router V2 grader family."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import StudyPrompt  # noqa: E402
from theriad.routing.study_adapters import (  # noqa: E402
    run_bfcl_smoke_fixtures,
    run_tau_environment_smoke,
)
from theriad.routing.study_graders import (  # noqa: E402
    TRUSTED_RESULT_SCHEMA_VERSION,
    canonical_hash,
    validate_pairwise_contract,
    verify_trusted_result,
)
from theriad.routing.study_grading import grade_response, randomize_pairwise_positions  # noqa: E402


def _base(prompt_id: str, label_type: str, grader: dict) -> StudyPrompt:
    return StudyPrompt(
        prompt_id=prompt_id,
        prompt="smoke fixture prompt",
        split="validation",
        source="router-v2-smoke",
        benchmark_family="router-v2-smoke",
        domain="smoke",
        template_family=f"smoke:{label_type}",
        collection_vintage="smoke-v1",
        license_name="internal",
        license_status="internal",
        source_revision="smoke-v1",
        provenance_url="internal://router-v2-smoke",
        lane="general",
        label_type=label_type,
        task_format="conversational_reasoning",
        leakage_group=f"smoke:{prompt_id}",
        grader=grader,
    )


def _trusted_fixture(kind: str, executor_id: str, executor_version: str) -> dict:
    evidence = {"fixture": kind, "passed_check": True}
    return {
        "schema_version": TRUSTED_RESULT_SCHEMA_VERSION,
        "prompt_id": f"smoke-{kind}",
        "result": True,
        "executor_id": executor_id,
        "executor_version": executor_version,
        "evidence": evidence,
        "evidence_sha256": canonical_hash(evidence),
    }


def run() -> dict:
    deterministic = _base(
        "smoke-deterministic",
        "deterministic",
        {"answer": "A", "answer_type": "multiple_choice"},
    )
    deterministic_grade = grade_response(deterministic, "ANSWER: A")

    bfcl_grader = {
        "kind": "bfcl_official_trusted_result",
        "executor_id": "bfcl-v4-checker",
        "executor_version": "bfcl-v4@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "evidence_policy": "sha256_trusted_result",
    }
    bfcl = _base("smoke-bfcl", "tool_state", bfcl_grader)
    bfcl_result = _trusted_fixture(
        "bfcl", bfcl_grader["executor_id"], bfcl_grader["executor_version"]
    )
    bfcl_result["prompt_id"] = bfcl.prompt_id
    bfcl_grade = verify_trusted_result(
        bfcl.grader, prompt_id=bfcl.prompt_id, result=bfcl_result
    )
    bfcl_response_grade = grade_response(bfcl, "ignored", trusted_result=bfcl_result)

    tau_grader = {
        "kind": "tau3_official_trusted_result",
        "executor_id": "tau2-evaluate-trajs",
        "executor_version": "tau3@a2c024725189473d2d7cea3a5cfdbcc67478e41f",
        "evidence_policy": "sha256_trusted_result",
    }
    tau = _base("smoke-tau3", "tool_state", tau_grader)
    tau_result = _trusted_fixture(
        "tau3", tau_grader["executor_id"], tau_grader["executor_version"]
    )
    tau_result["prompt_id"] = tau.prompt_id
    tau_grade = verify_trusted_result(
        tau.grader, prompt_id=tau.prompt_id, result=tau_result
    )
    pending_tau = grade_response(tau, "ignored")

    pairwise_grader = {
        "kind": "frozen_pairwise_judge",
        "judge_id": "router-v2-local-judge",
        "judge_version": "router-v2-pairwise-v2",
        "criteria_version": "pairwise-criteria-v1",
        "position_randomization": "prompt_id_seeded_sha256",
        "sampling_settings": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
        "tie_policy": "record_tie_without_forcing_binary_label",
        "abstention_policy": "judge_may_abstain_on rubric_or_context_failure",
        "disagreement_review_policy": "independent_manual_review_then_adjudication; no automatic overwrite",
        "judging_cost_status": "postponed_unpriced",
        "expected_judging_cost_usd": None,
        "worst_case_judging_cost_usd": None,
    }
    pairwise = _base("smoke-pairwise", "pairwise", pairwise_grader)
    validate_pairwise_contract(pairwise.grader)
    presentation = randomize_pairwise_positions(
        pairwise.prompt_id, "reference", "challenger", judge_version="router-v2-pairwise-v2"
    )

    return {
        "status": "passed_with_executor_gaps",
        "provider_calls": 0,
        "families": {
            "deterministic": {
                "status": deterministic_grade.status,
                "label": deterministic_grade.label,
            },
            "bfcl_trusted_result": {
                "status": bfcl_response_grade.status,
                "label": bfcl_response_grade.label,
                "evidence_sha256": bfcl_grade["evidence_sha256"],
            },
            "tau3_trusted_result": {
                "verified_fixture_label": tau_grade["label"],
                "missing_live_state_status": pending_tau.status,
            },
            "pairwise": {
                "status": "contract_and_position_smoke_passed",
                "judge_version": pairwise.grader["judge_version"],
                "left_is_reference": presentation.left_is_reference,
                "live_judging": "not_attempted",
            },
            "bfcl_fixtures": run_bfcl_smoke_fixtures(),
            "tau_bench_environment": run_tau_environment_smoke(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
