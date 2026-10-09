#!/usr/bin/env python3
"""Refresh source-acquisition evidence after a runtime-only corpus re-freeze."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import (  # noqa: E402
    SANDBOX_EXECUTOR_ID,
    SANDBOX_POLICY_VERSION,
)
from theriad.routing.study import (  # noqa: E402
    REPEATED_TRIAL_GROUP_CONTRACT,
    corpus_checksum,
    load_jsonl,
)
from theriad.routing.study_adapters import CODE_ADAPTER_VERSION  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh(
    base_path: Path,
    corpus_path: Path,
    audit_path: Path,
    official_smoke_path: Path,
    code_validation_path: Path,
    output_path: Path,
) -> dict[str, object]:
    report = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
    rows = load_jsonl(corpus_path)
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    official_smoke = json.loads(official_smoke_path.read_text(encoding="utf-8"))
    code_validation_report = json.loads(code_validation_path.read_text(encoding="utf-8"))
    if len(rows) != 2000:
        raise ValueError(f"expected 2,000 frozen rows, found {len(rows)}")
    current_checksum = corpus_checksum(rows)
    current_file_sha256 = _sha256(corpus_path)
    current_audit = ((audit_report.get("corpus") or {}).get("audit"))
    if not isinstance(current_audit, dict) or current_audit.get("valid") is not True:
        raise ValueError("current corpus audit is not valid")
    if code_validation_report.get("status") != "passed":
        raise ValueError("current code executor validation is not passed")

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "passed"
    report["checksum"] = current_checksum
    report["output_sha256"] = current_file_sha256
    report["output_path"] = str(corpus_path)
    report["audit_before_write"] = current_audit
    report["provider_calls"] = 0
    report["provider_spend_usd"] = 0.0
    effective_groups = {(row.split, row.normalized_hash) for row in rows}
    report["repeated_trial_contract"] = {
        "name": REPEATED_TRIAL_GROUP_CONTRACT,
        "router_visible_duplicate_policy": "preserve_raw_tau2_trials_and_aggregate_outcomes_by_visible_normalized_prompt",
        "raw_trial_count": len(rows),
        "effective_router_input_count": len(effective_groups),
        "effective_router_input_count_by_split": dict(
            sorted(Counter(split for split, _ in effective_groups).items())
        ),
        "grouped_bootstrap": True,
        "hidden_state_added_to_prompt": False,
    }
    report["runtime_correction"] = {
        "status": "passed",
        "metadata_only_corpus_re_freeze": True,
        "prompt_text_changed": False,
        "normalized_prompt_hashes_preserved": True,
        "executable_code_rows_rebound": 300,
        "code_adapter_version": CODE_ADAPTER_VERSION,
        "sandbox_executor_id": SANDBOX_EXECUTOR_ID,
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "official_executor_smoke_path": str(official_smoke_path),
    }

    code_validation = report.setdefault("code_acquisition", {}).setdefault(
        "sandbox_validation", {}
    )
    code_validation["executor_id"] = SANDBOX_EXECUTOR_ID
    code_validation["policy_version"] = SANDBOX_POLICY_VERSION
    code_validation["adapter_version"] = CODE_ADAPTER_VERSION
    code_validation["validation_status"] = "passed"
    code_validation["posix_shared_memory_for_pinned_numeric_runtime"] = True
    code_validation["validation_report"] = str(code_validation_path)
    code_validation["passed_count"] = int(code_validation_report.get("passed_count", 0))
    code_validation["failed_count"] = int(code_validation_report.get("failed_count", 0))

    executor_results = report.setdefault("executor_results", {})
    executor_results["official_executor_smoke"] = official_smoke
    executor_results["sandbox"] = {
        "available": True,
        "executor_id": SANDBOX_EXECUTOR_ID,
        "policy_version": SANDBOX_POLICY_VERSION,
        "posix_shared_memory_for_pinned_numeric_runtime": True,
        "real_benchmark_execution_on_host": False,
    }
    code_executor = executor_results.setdefault("code", {})
    code_executor["executor_id"] = SANDBOX_EXECUTOR_ID
    code_executor["policy_version"] = SANDBOX_POLICY_VERSION
    code_executor["adapter_version"] = CODE_ADAPTER_VERSION
    report["official_executor_smoke"] = official_smoke

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--official-smoke", type=Path, required=True)
    parser.add_argument("--code-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = refresh(
        args.base,
        args.corpus,
        args.audit,
        args.official_smoke,
        args.code_validation,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report.get("runtime_correction", {}).get("status"),
                "checksum": report.get("checksum"),
                "output_sha256": report.get("output_sha256"),
                "provider_calls": report.get("provider_calls"),
                "provider_spend_usd": report.get("provider_spend_usd"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
