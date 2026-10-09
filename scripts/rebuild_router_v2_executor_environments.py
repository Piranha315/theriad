#!/usr/bin/env python3
"""Rebuild all Router V2 grader environments in a disposable directory.

This is provider-free.  The temporary environments are created from the
recorded interpreter identities, normalized dependency locks, and pinned local
source revisions.  They are smoke-tested inside the same macOS sandbox and
deleted before the report is returned.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_router_v2_official_executors import run as run_official_smoke  # noqa: E402
from theriad.routing.executor_manifest import (  # noqa: E402
    ExecutorManifestError,
    create_executor_manifest,
    load_executor_manifest,
    reproducible_contract_hash,
)
from theriad.routing.sandbox import (  # noqa: E402
    default_policy,
    run_python_tests,
)


def _run(command: tuple[str, ...]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"rebuild command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr[-4000:]
        )
    return {"command": list(command), "returncode": result.returncode, "stdout": result.stdout[-2000:]}


def _build_runtime(
    *,
    role: str,
    base_python: Path,
    target_root: Path,
    lock_path: Path,
    source: Path | None,
) -> dict[str, Any]:
    target = target_root / role
    target_python = target / "bin/python"
    steps: list[dict[str, Any]] = []
    steps.append(_run(("uv", "venv", "--python", str(base_python), str(target))))
    steps.append(_run(("uv", "pip", "sync", "--python", str(target_python), str(lock_path))))
    if source is not None:
        steps.append(
            _run(
                (
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(target_python),
                    "--no-deps",
                    str(source),
                )
            )
        )
    return {"role": role, "python": str(target_python), "steps": steps}


def run(*, manifest_path: Path, output: Path, source_cache: Path) -> dict[str, Any]:
    manifest = load_executor_manifest(manifest_path)
    acquired = source_cache / "acquired"
    bfcl_source = acquired / "bfcl/berkeley-function-call-leaderboard"
    bfcl_cache = source_cache / "bfcl-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
    tau_source = acquired / "tau2-bench"
    code_cache = acquired / "code-data"
    with tempfile.TemporaryDirectory(prefix="theriad-router-v2-rebuild-") as temporary:
        rebuild_root = Path(temporary)
        built: dict[str, Any] = {}
        for role in ("code_grader", "bfcl", "tau2"):
            record = manifest["runtimes"][role]
            source = None
            if role == "bfcl":
                source = bfcl_source
            elif role == "tau2":
                source = tau_source
            built[role] = _build_runtime(
                role=role,
                base_python=Path(str(record["resolved_executable"])),
                target_root=rebuild_root,
                lock_path=Path(str(record["dependency_lock_path"])),
                source=source,
            )

        code_python = Path(built["code_grader"]["python"])
        code_smoke = run_python_tests(
            "def answer(value):\n    return value + 1\n",
            "def check(function):\n    assert function(41) == 42\n",
            entry_point="answer",
            policy=default_policy(runtime=str(code_python), allowed_read_paths=(ROOT,)),
        )
        official = run_official_smoke(
            bfcl_python=Path(built["bfcl"]["python"]),
            bfcl_cache=bfcl_cache,
            bfcl_source=bfcl_source,
            tau_python=Path(built["tau2"]["python"]),
            tau_source=tau_source,
        )

        fresh_manifest_path = rebuild_root / "manifest.json"
        fresh_manifest = create_executor_manifest(
            output=fresh_manifest_path,
            source_cache=source_cache,
            code_python=code_python,
            bfcl_python=Path(built["bfcl"]["python"]),
            tau_python=Path(built["tau2"]["python"]),
            code_source=ROOT,
            code_cache=code_cache,
            bfcl_source=bfcl_source,
            bfcl_cache=bfcl_cache,
            tau_source=tau_source,
            expected_bfcl_revision=str(manifest["runtimes"]["bfcl"]["source_revision"]),
            expected_tau_revision=str(manifest["runtimes"]["tau2"]["source_revision"]),
        )
        load_executor_manifest(fresh_manifest_path)
        contract_checks: dict[str, Any] = {}
        for role in ("code_grader", "bfcl", "tau2"):
            expected = manifest["runtimes"][role]
            actual = fresh_manifest["runtimes"][role]
            expected_contract = str(expected.get("reproducible_contract_hash") or "")
            actual_contract = reproducible_contract_hash(actual)
            contract_checks[role] = {
                "expected_reproducible_contract_hash": expected_contract,
                "actual_reproducible_contract_hash": actual_contract,
                "match": expected_contract == actual_contract,
                "dependency_lock_sha256_match": actual.get("dependency_lock_sha256")
                == expected.get("dependency_lock_sha256"),
                "source_revision_match": actual.get("source_revision")
                == expected.get("source_revision"),
            }
            if not all(
                (
                    contract_checks[role]["match"],
                    contract_checks[role]["dependency_lock_sha256_match"],
                    contract_checks[role]["source_revision_match"],
                )
            ):
                raise RuntimeError(f"fresh {role} environment does not match its lock contract")

        report = {
            "schema_version": 1,
            "status": "passed",
            "sys_version": sys.version,
            "sys_executable": sys.executable,
            "resolved_executable": os.path.realpath(sys.executable),
            "platform": platform.platform(),
            "sandbox_policy_version": official["sandbox"]["policy_version"],
            "exact_test_command": shlex.join(sys.argv),
            "grader_runtime_selection": official["grader_runtime_selection"],
            "manifest_path": str(manifest_path),
            "source_cache": str(source_cache),
            "fresh_rebuild_directory": str(rebuild_root),
            "fresh_rebuild_directory_removed": False,
            "builds": built,
            "code_smoke": code_smoke.to_dict(),
            "official_executor_smoke": official,
            "fresh_manifest_hash": fresh_manifest["manifest_hash"],
            "contract_checks": contract_checks,
            "provider_calls": 0,
            "provider_spend_usd": 0.0,
        }
    report["fresh_rebuild_directory_removed"] = not rebuild_root.exists()
    if rebuild_root.exists():
        shutil.rmtree(rebuild_root)
        report["fresh_rebuild_directory_removed"] = not rebuild_root.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reports/router-v2/router-v2-executor-environment-manifest.json",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("/tmp/theriad-router-v2-source-cache"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(
            manifest_path=args.manifest,
            output=args.output,
            source_cache=args.source_cache,
        )
    except (ExecutorManifestError, OSError, RuntimeError, ValueError) as exc:
        print(f"router-v2 executor rebuild stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
