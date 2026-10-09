#!/usr/bin/env python3
"""Run pinned BFCL and tau2 official executor smokes in the local sandbox.

This command never contacts a provider.  The benchmark checker/environment
packages run only through the Router V2 macOS sandbox; a missing sandbox or a
non-zero checker result is a hard failure.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import (  # noqa: E402
    SandboxResult,
    default_policy,
    run_isolated,
    runtime_details,
    sandbox_available,
)

BFCL_CHECKER_ID = "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker"
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TAU_EVALUATOR_ID = "tau2.evaluator.evaluator.evaluate_simulation"
TAU_REVISION = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"


def _run_sandboxed_python(
    python: Path, code: str, *, allowed_read_paths: tuple[Path, ...]
) -> tuple[SandboxResult, dict]:
    with tempfile.TemporaryDirectory(prefix="theriad-router-v2-executor-") as directory:
        result = run_isolated(
            (str(python), "-c", code),
            task_directory=Path(directory),
            policy=replace(
                default_policy(
                    runtime=str(python),
                    allowed_read_paths=tuple(str(path) for path in allowed_read_paths),
                ),
                cpu_seconds=30,
                wall_clock_seconds=30.0,
            ),
        )
    if not result.passed:
        raise RuntimeError(
            f"official executor smoke failed: status={result.status} "
            f"returncode={result.returncode} stderr={result.stderr[-2000:]}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"official executor smoke did not emit JSON: {result.stdout!r}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError(f"official executor smoke returned an invalid result: {payload!r}")
    return result, payload


def _bfcl_code(cache: Path, source: Path, site_packages: Path) -> str:
    return f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(site_packages)!r})
sys.path.insert(0, {str(source)!r})
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_relevance_entry
from bfcl_eval.model_handler.api_inference.gorilla import GorillaHandler

base = Path({str(cache)!r})

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def first(path, item_id):
    for row in rows(path):
        if str(row.get("id")) == item_id:
            return row
    raise AssertionError(f"missing fixture {{item_id}} in {{path}}")

def pick(value):
    if isinstance(value, list):
        for item in value:
            if item != "":
                return item
        return ""
    return value

def candidate_from_ground_truth(ground_truth):
    candidate = []
    for item in ground_truth:
        if not item:
            candidate.append({{}})
            continue
        for name, arguments in item.items():
            candidate.append({{name: {{key: pick(value) for key, value in arguments.items()}}}})
    return candidate

fixture_results = {{}}
for stem, category in (("simple_python", "simple_python"), ("multiple", "multiple"), ("parallel", "parallel")):
    task = first(base / f"BFCL_v4_{{stem}}.json", f"{{stem}}_0")
    answer = first(base / "possible_answer" / f"BFCL_v4_{{stem}}.json", f"{{stem}}_0")
    ground_truth = answer["ground_truth"]
    candidate = candidate_from_ground_truth(ground_truth)
    checked = ast_checker(
        task["function"], candidate, ground_truth, Language.PYTHON, category,
        "gorilla-openfunctions-v2",
    )
    if not checked.get("valid"):
        raise AssertionError(f"official BFCL {{category}} checker rejected its own fixture: {{checked}}")
    fixture_results[category] = {{"passed": True, "ground_truth_call_count": len(ground_truth)}}

task = first(base / "BFCL_v4_irrelevance.json", "irrelevance_0")
handler = GorillaHandler("gorilla-openfunctions-v2", 0.0, "gorilla-openfunctions-v2", False)
checked = _evaluate_single_relevance_entry(
    handler, task["id"], "", task, "gorilla-openfunctions-v2", "irrelevance"
)
if not checked.get("valid"):
    raise AssertionError(f"official BFCL irrelevance checker rejected empty-call fixture: {{checked}}")
fixture_results["irrelevance"] = {{"passed": True, "candidate_call_count": 0}}

print(json.dumps({{
    "status": "passed",
    "checker_id": "{BFCL_CHECKER_ID}",
    "fixtures": fixture_results,
    "ground_truth_sequence_preserved": True,
    "provider_calls": 0,
}}))
"""


def _tau_code(source: Path, site_packages: Path) -> str:
    return f"""
import json
import sys

sys.path.insert(0, {str(site_packages)!r})
sys.path.insert(0, {str(source / 'src')!r})
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.domains.mock.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.run import get_tasks

task = get_tasks("mock", task_ids=["create_task_1"])[0]
environment = get_environment()
call = ToolCall(
    id="call-1",
    name="create_task",
    arguments={{"user_id": "user_1", "title": "Important Meeting"}},
)
tool_result = environment.get_response(call)
simulation = SimulationRun(
    id="official-smoke-create-task",
    task_id=task.id,
    start_time="2026-01-01T00:00:00",
    end_time="2026-01-01T00:00:01",
    duration=1.0,
    termination_reason=TerminationReason.AGENT_STOP,
    messages=[AssistantMessage.text("", tool_calls=[call]), tool_result],
)
reward_info = evaluate_simulation(
    simulation, task, EvaluationType.ENV, False, "mock", strict_replay=False
)
db_match = bool(reward_info.db_check and reward_info.db_check.db_match)
if reward_info.reward != 1.0 or not db_match:
    raise AssertionError({{"reward": reward_info.reward, "db_match": db_match}})
print(json.dumps({{
    "status": "passed",
    "evaluator_id": "{TAU_EVALUATOR_ID}",
    "task_id": task.id,
    "interactive_tool_loop": True,
    "persisted_environment_state": True,
    "reward": reward_info.reward,
    "db_match": db_match,
    "provider_calls": 0,
}}))
"""


def run(
    *,
    bfcl_python: Path,
    bfcl_cache: Path,
    bfcl_source: Path,
    tau_python: Path,
    tau_source: Path,
) -> dict:
    if not sandbox_available():
        raise RuntimeError("approved macOS sandbox-exec backend is unavailable")
    for path in (bfcl_python, bfcl_cache, bfcl_source, tau_python, tau_source):
        if not path.exists():
            raise FileNotFoundError(path)

    bfcl_site_packages = next((path for path in (bfcl_python.parent.parent / "lib").glob("python*/site-packages") if path.is_dir()), None)
    tau_site_packages = next((path for path in (tau_python.parent.parent / "lib").glob("python*/site-packages") if path.is_dir()), None)
    if bfcl_site_packages is None or tau_site_packages is None:
        raise RuntimeError("could not locate pinned virtual-environment site-packages")
    bfcl_result, bfcl_payload = _run_sandboxed_python(
        bfcl_python,
        _bfcl_code(bfcl_cache.resolve(), bfcl_source.resolve(), bfcl_site_packages.resolve()),
        allowed_read_paths=(bfcl_cache, bfcl_source, bfcl_site_packages),
    )
    tau_result, tau_payload = _run_sandboxed_python(
        tau_python,
        _tau_code(tau_source.resolve(), tau_site_packages.resolve()),
        allowed_read_paths=(tau_source, tau_site_packages),
    )
    host_runtime = runtime_details()
    return {
        "schema_version": 1,
        "status": "passed",
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "resolved_executable": host_runtime["resolved_executable"],
        "platform": platform.platform(),
        "exact_test_command": shlex.join([sys.executable, *sys.argv]),
        "grader_runtime_selection": {
            "bfcl": runtime_details(bfcl_python),
            "tau2": runtime_details(tau_python),
        },
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "sandbox": {
            "available": True,
            "executor_id": bfcl_result.executor_id,
            "policy_version": bfcl_result.policy_version,
            "network_disabled": True,
            "base_filesystem_read_only": True,
            "task_directory_only_writable": True,
            "posix_shared_memory_for_pinned_numeric_runtime": True,
            "host_code_execution": False,
            "credentials_in_environment": False,
        },
        "bfcl": {
            "status": "passed",
            "revision": BFCL_REVISION,
            "python": str(bfcl_python),
            "source": str(bfcl_source),
            "cache": str(bfcl_cache),
            "official_checker": bfcl_payload,
            "sandbox_result": bfcl_result.to_dict(),
        },
        "tau2": {
            "status": "passed",
            "revision": TAU_REVISION,
            "python": str(tau_python),
            "source": str(tau_source),
            "official_environment": tau_payload,
            "sandbox_result": tau_result.to_dict(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-python", type=Path, required=True)
    parser.add_argument("--bfcl-cache", type=Path, required=True)
    parser.add_argument("--bfcl-source", type=Path, required=True)
    parser.add_argument("--tau-python", type=Path, required=True)
    parser.add_argument("--tau-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        bfcl_python=args.bfcl_python,
        bfcl_cache=args.bfcl_cache,
        bfcl_source=args.bfcl_source,
        tau_python=args.tau_python,
        tau_source=args.tau_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
