#!/usr/bin/env python3
"""Record the active Router V2 runtime identity and sandbox regression commands."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import (  # noqa: E402
    SANDBOX_EXECUTOR_ID,
    SANDBOX_POLICY_VERSION,
    SUPPORTED_RUNTIME_APPROACH,
    default_policy,
    runtime_details,
)


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv/bin/python",
        help="the exact project runtime to validate; no fallback is selected",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    python = args.python if args.python.is_absolute() else ROOT / args.python
    pytest = python.parent / "pytest"
    commands = [
        [str(python), "-VV"],
        [
            str(python),
            "-c",
            "import os,sys; print(sys.executable); print(os.path.realpath(sys.executable))",
        ],
        [str(pytest), "-q", "tests/unit/routing/test_sandbox_adversarial.py"],
        [
            str(pytest),
            "-q",
            "tests/unit/routing/test_study_runtime.py::test_executable_code_adapter_uses_real_sandbox_and_never_plain_grader",
        ],
    ]
    results = [_run(command) for command in commands]
    runtime = runtime_details(str(python))
    policy = default_policy(runtime=str(python))
    report = {
        "schema_version": 1,
        "status": "passed" if all(item["returncode"] == 0 for item in results) else "blocked",
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "runtime": runtime,
        "sys_version": runtime["sys_version"],
        "sys_executable": runtime["sys_executable"],
        "resolved_executable": runtime["resolved_executable"],
        "platform": runtime["platform"],
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "exact_test_commands": [item["command"] for item in results],
        "grader_runtime_selection": {
            "approach": SUPPORTED_RUNTIME_APPROACH,
            "requested_executable": str(python),
            "resolved_executable": runtime["resolved_executable"],
            "python_app_helper": runtime["python_app_helper"],
            "process_exec_allowlist": runtime["process_exec_allowlist"],
            "supported": runtime["supported"],
            "reasons": runtime["reasons"],
        },
        "sandbox": {
            "executor_id": SANDBOX_EXECUTOR_ID,
            "policy_version": SANDBOX_POLICY_VERSION,
            "network_disabled": policy.network_disabled,
            "base_filesystem_read_only": policy.base_filesystem_read_only,
            "task_directory_only_writable": policy.task_directory_only_writable,
            "credentials_in_environment": policy.credentials_in_environment,
            "cpu_seconds": policy.cpu_seconds,
            "memory_bytes": policy.memory_bytes,
            "process_limit": policy.process_limit,
            "wall_clock_seconds": policy.wall_clock_seconds,
            "output_bytes": policy.output_bytes,
            "runtime_allowlist": list(policy.runtime_allowlist),
        },
        "exact_commands": results,
        "prior_independent_failure_observation": {
            "sandbox_command": f"{pytest} -q tests/unit/routing/test_sandbox_adversarial.py",
            "sandbox_returncode": 1,
            "sandbox_summary": "4 failed under Python 3.14.6 with posix_spawn Undefined error: 0",
            "code_adapter_command": f"{pytest} -q tests/unit/routing/test_study_runtime.py::test_executable_code_adapter_uses_real_sandbox_and_never_plain_grader",
            "code_adapter_returncode": 1,
            "code_adapter_summary": "correct fixture response received grade=False before the runtime path fix",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "provider_calls", "provider_spend_usd")}, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
