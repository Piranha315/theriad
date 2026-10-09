#!/usr/bin/env python3
"""Validate accepted code-task harnesses through the isolated sandbox only."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import (  # noqa: E402
    CODE_GRADER_RUNTIME_ENV,
    DEFAULT_CPU_SECONDS,
    DEFAULT_MEMORY_BYTES,
    DEFAULT_OUTPUT_BYTES,
    DEFAULT_PROCESS_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    SANDBOX_EXECUTOR_ID,
    SANDBOX_POLICY_VERSION,
    SUPPORTED_RUNTIME_APPROACH,
    SandboxUnavailable,
    default_policy,
    run_python_tests,
    runtime_details,
    sandbox_available,
)
from theriad.routing.study import load_jsonl  # noqa: E402

CODE_FILES = {
    "bigcode/bigcodebench": "bigcodebench/v0.1.4-00000-of-00001.parquet",
    "evalplus/humanevalplus": "humanevalplus/test-00000-of-00001-5973903632b82d40.parquet",
    "evalplus/mbppplus": "mbppplus/test-00000-of-00001-d5781c9c51e02795.parquet",
}


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return [dict(row) for row in pd.read_parquet(path).to_dict("records")]


def _source(row: dict[str, Any], source: str) -> str:
    if source == "bigcode/bigcodebench":
        return _text(row.get("code_prompt")) + "\n" + _text(row.get("canonical_solution"))
    if source == "evalplus/mbppplus":
        return _text(row.get("code"))
    return _text(row.get("prompt")) + "\n" + _text(row.get("canonical_solution"))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(value or "")


def _runtime_probe(runtime: str | None) -> dict[str, Any]:
    if not runtime:
        return {"status": "missing", "sys_version": None, "sys_executable": None}
    command = (
        "import os, sys; "
        "print(sys.version); "
        "print(sys.executable); "
        "print(os.path.realpath(sys.executable))"
    )
    try:
        completed = subprocess.run(
            [runtime, "-c", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "error": str(exc)}
    lines = completed.stdout.splitlines()
    return {
        "status": "passed" if completed.returncode == 0 and len(lines) >= 3 else "failed",
        "returncode": completed.returncode,
        "sys_version": lines[0] if lines else None,
        "sys_executable": lines[1] if len(lines) > 1 else None,
        "resolved_executable": lines[2] if len(lines) > 2 else None,
        "stderr": completed.stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--code-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--grader-python",
        type=str,
        default=None,
        help=f"pinned runtime; defaults to {CODE_GRADER_RUNTIME_ENV}",
    )
    args = parser.parse_args()

    configured_grader_runtime = (
        str(args.grader_python)
        if args.grader_python is not None
        else os.environ.get(CODE_GRADER_RUNTIME_ENV, "").strip()
    )
    host_runtime = runtime_details()
    runtime = (
        runtime_details(configured_grader_runtime)
        if configured_grader_runtime
        else {
            "approach": SUPPORTED_RUNTIME_APPROACH,
            "selection": "missing_required_pinned_runtime",
            "configured_runtime_env": CODE_GRADER_RUNTIME_ENV,
            "requested_executable": None,
            "resolved_executable": None,
            "python_app_helper": None,
            "python_home": None,
            "platform": platform.platform(),
            "platform_system": sys.platform,
            "process_exec_allowlist": [],
            "sandbox_exec": None,
            "supported": False,
            "reasons": [f"{CODE_GRADER_RUNTIME_ENV} is not configured"],
        }
    )
    runtime_probe = _runtime_probe(configured_grader_runtime or None)
    exact_test_command = shlex.join([sys.executable, *sys.argv])
    rows = [row for row in load_jsonl(args.corpus) if row.task_format == "code_execution"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "resolved_executable": runtime["resolved_executable"],
        "platform": platform.platform(),
        "exact_test_command": exact_test_command,
        "host_runtime": host_runtime,
        "grader_runtime_selection": {
            "approach": SUPPORTED_RUNTIME_APPROACH,
            "environment_variable": CODE_GRADER_RUNTIME_ENV,
            "configured": bool(configured_grader_runtime),
            "current_interpreter_must_match": True,
            "requested_executable": runtime["requested_executable"],
            "resolved_executable": runtime["resolved_executable"],
            "python_app_helper": runtime["python_app_helper"],
            "process_exec_allowlist": runtime["process_exec_allowlist"],
            "supported": runtime["supported"],
            "reasons": runtime["reasons"],
            "runtime_identity": runtime_probe,
        },
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "sandbox": {
            "available": sandbox_available(),
            "executor_id": SANDBOX_EXECUTOR_ID,
            "policy_version": SANDBOX_POLICY_VERSION,
            "network_disabled": True,
            "base_filesystem_read_only": True,
            "task_directory_only_writable": True,
            "credentials_in_environment": False,
            "posix_shared_memory_for_runtime": True,
            "cpu_seconds": DEFAULT_CPU_SECONDS,
            "memory_bytes": DEFAULT_MEMORY_BYTES,
            "process_limit": DEFAULT_PROCESS_LIMIT,
            "wall_clock_seconds": DEFAULT_TIMEOUT_SECONDS,
            "output_bytes": DEFAULT_OUTPUT_BYTES,
            "runtime": runtime["resolved_executable"],
            "runtime_details": runtime,
            "host_code_execution": False,
        },
        "accepted_code_rows": len(rows),
        "results": [],
    }
    current_resolved = str(Path(sys.executable).resolve())
    selected_resolved = runtime.get("resolved_executable")
    if not configured_grader_runtime:
        report["status"] = "blocked_missing_pinned_grader_runtime"
        report["reason"] = f"set {CODE_GRADER_RUNTIME_ENV} or pass --grader-python"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2
    if selected_resolved != current_resolved:
        report["status"] = "blocked_runtime_mismatch"
        report["reason"] = (
            "run the validator with the pinned grader runtime; host interpreter "
            f"{current_resolved!r} differs from selected {selected_resolved!r}"
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2
    if not sandbox_available() or runtime["supported"] is not True:
        report["status"] = (
            "blocked_sandbox_unavailable"
            if not sandbox_available()
            else "blocked_runtime_unavailable"
        )
        report["reason"] = "; ".join(
            [
                "refusing to execute benchmark code on the host",
                *[str(reason) for reason in runtime["reasons"]],
            ]
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source, relative in CODE_FILES.items():
        for raw in _read_parquet(args.code_cache / relative):
            source_rows[(source, str(raw.get("task_id")))] = raw
    failures: list[dict[str, Any]] = []
    try:
        policy = default_policy(runtime=configured_grader_runtime)
        for row in rows:
            grader = row.grader
            key = (row.source, str(grader.get("source_task_id")))
            raw = source_rows.get(key)
            if raw is None:
                failures.append({"prompt_id": row.prompt_id, "reason": "source_task_not_found"})
                continue
            result = run_python_tests(
                _source(raw, row.source),
                str(grader["executable_tests"]),
                entry_point=str(grader["entry_point"]),
                policy=policy,
            )
            item = {
                "prompt_id": row.prompt_id,
                "source": row.source,
                "source_task_id": str(grader.get("source_task_id")),
                **result.to_dict(),
            }
            report["results"].append(item)
            if not result.passed:
                failures.append(item)
    except SandboxUnavailable as exc:
        report["status"] = "blocked_sandbox_unavailable"
        report["reason"] = str(exc)
        failures.append({"reason": str(exc)})

    report["passed_count"] = len(rows) - len(failures)
    report["failed_count"] = len(failures)
    report["failures"] = failures[:50]
    report["status"] = "passed" if not failures else "blocked_code_executor_failures"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "accepted_code_rows", "passed_count", "failed_count")}, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
