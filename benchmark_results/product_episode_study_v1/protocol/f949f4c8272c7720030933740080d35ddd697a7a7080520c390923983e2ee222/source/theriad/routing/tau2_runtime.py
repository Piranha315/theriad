"""Pinned, provider-free tau2 environment bridge for Router V2.

The tau2 package is kept outside the application interpreter. Each environment
operation and official evaluation runs in the pinned tau2 virtualenv through
the same sandbox used for executable code.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .sandbox import default_policy, run_isolated, sandbox_available

TAU2_REVISION = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
TAU2_EVALUATOR_ID = "tau2.evaluator.evaluator.evaluate_simulation"


class Tau2Unavailable(RuntimeError):
    """Raised when the pinned tau2 runtime cannot be admitted safely."""


def _paths() -> tuple[Path, Path]:
    root = Path(
        os.environ.get("THERIAD_ROUTER_V2_SOURCE_CACHE", "/tmp/theriad-router-v2-source-cache")
    )
    return (
        Path(os.environ.get("THERIAD_TAU2_PYTHON", str(root / "acquired/tau2-venv/bin/python"))),
        Path(os.environ.get("THERIAD_TAU2_SOURCE", str(root / "acquired/tau2-bench"))),
    )


def tau2_available() -> bool:
    python, source = _paths()
    return sandbox_available() and python.is_file() and source.is_dir()


def _site_packages(python: Path) -> Path | None:
    return next(
        (
            path
            for path in (python.parent.parent / "lib").glob("python*/site-packages")
            if path.is_dir()
        ),
        None,
    )


def _runner_paths(python: Path, source: Path) -> tuple[str, str, tuple[str, ...]]:
    source_path = source.resolve()
    site = _site_packages(python)
    target_python = python.resolve()
    target_root = target_python.parent.parent
    version = target_python.parts[-2].removeprefix("python")
    stdlib = target_root / "lib" / f"python{version}"
    read_paths = tuple(
        str(path.resolve())
        for path in (
            source_path,
            source_path / "src",
            site,
            python.parent.parent,
            target_root,
            stdlib,
            stdlib / "lib-dynload",
        )
        if path is not None and path.exists()
    )
    return str(source_path), str(site.resolve()) if site else "", read_paths


class PinnedTau2Environment:
    """A replayable official tau2 domain environment."""

    def __init__(self, row: Any) -> None:
        self.python, self.source = _paths()
        if not tau2_available():
            raise Tau2Unavailable("pinned tau2 runtime or sandbox is unavailable")
        grader = row.grader
        self.task_id = str(grader.get("official_task_id") or "")
        self.task_set = str(grader.get("official_task_set") or row.domain or "")
        if not self.task_id or not self.task_set:
            raise Tau2Unavailable("tau2 row lacks an official task id/task set")
        self._calls: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {
            "task_id": self.task_id,
            "task_set": self.task_set,
            "tool_call_count": 0,
            "state_identity": grader.get("state_identity"),
        }
        self._schemas: tuple[dict[str, Any], ...] | None = None

    def reset(self, initial_state: Mapping[str, Any]) -> None:
        self._calls = []
        self._state["reset_initial_state_supplied"] = bool(initial_state)
        self._state["tool_call_count"] = 0

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("task_id") not in {None, self.task_id}:
            raise Tau2Unavailable("resume state belongs to another tau2 task")
        self._state.update(dict(state))

    def _code_header(self) -> str:
        source, site, _ = _runner_paths(self.python, self.source)
        return (
            "import json, os, sys\n"
            # The sandbox intentionally denies network access.  Force
            # LiteLLM to use its pinned local cost map so importing the
            # official tau2 evaluator cannot spend most of the wall-clock
            # budget attempting a remote metadata fetch.
            "os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'\n"
            f"sys.path.insert(0, {site!r})\n"
            f"sys.path.insert(0, {str(Path(source) / 'src')!r})\n"
            "payload = json.loads(open('tau2-payload.json').read())\n"
        )

    def _run(self, body: str, payload: Mapping[str, Any]) -> str:
        _, _, read_paths = _runner_paths(self.python, self.source)
        with TemporaryDirectory(prefix="theriad-router-v2-tau2-", dir="/tmp") as directory:
            task_directory = Path(directory)
            (task_directory / "tau2-payload.json").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            result = run_isolated(
                (str(self.python), "-c", self._code_header() + body),
                task_directory=task_directory,
                policy=replace(
                    default_policy(runtime=str(self.python), allowed_read_paths=read_paths),
                    cpu_seconds=30,
                    wall_clock_seconds=30.0,
                ),
            )
        if not result.passed:
            raise Tau2Unavailable(
                f"official tau2 executor failed: {result.status}: {result.stderr[-2000:]}"
            )
        return result.stdout

    def tool_schemas(self) -> tuple[dict[str, Any], ...]:
        if self._schemas is not None:
            return self._schemas
        payload = {"task_set": self.task_set, "task_id": self.task_id}
        body = """
from tau2.registry import registry
from tau2.run import get_tasks

tasks = get_tasks(payload["task_set"], task_ids=[payload["task_id"]])
if len(tasks) != 1:
    raise RuntimeError("official tau2 task was not found")
domain = registry.get_env_constructor(payload["task_set"])()
print(json.dumps([tool.openai_schema for tool in domain.get_tools()]))
"""
        try:
            raw = json.loads(self._run(body, payload))
            self._schemas = tuple(dict(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise Tau2Unavailable("official tau2 tool schema output was invalid") from exc
        return self._schemas

    def apply_actions(self, actions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> None:
        for action in actions:
            self.apply_tool_call(
                type(
                    "Call",
                    (),
                    {
                        "name": str(action.get("name") or ""),
                        "arguments": dict(action.get("arguments") or {}),
                        "call_id": str(action.get("call_id") or ""),
                    },
                )()
            )

    def apply_tool_call(self, call: Any) -> Mapping[str, Any]:
        record = {
            "id": str(getattr(call, "call_id", "") or f"router-v2-{len(self._calls)}"),
            "name": str(getattr(call, "name", "")),
            "arguments": dict(getattr(call, "arguments", {}) or {}),
            "requestor": "assistant",
        }
        self._calls.append(record)
        payload = {"task_set": self.task_set, "task_id": self.task_id, "calls": self._calls}
        body = """
from tau2.data_model.message import ToolCall
from tau2.registry import registry
from tau2.run import get_tasks

task = get_tasks(payload["task_set"], task_ids=[payload["task_id"]])[0]
domain = registry.get_env_constructor(payload["task_set"])()
initial = task.initial_state
if initial is not None:
    domain.set_state(initial.initialization_data, initial.initialization_actions, initial.message_history or [])
last = None
for raw in payload["calls"]:
    last = domain.get_response(ToolCall(**raw))
print(json.dumps({
    "tool_message": last.model_dump(mode="json") if last is not None else None,
    "db_hash": domain.get_db_hash(),
    "user_db_hash": domain.get_user_db_hash(),
    "tool_call_count": len(payload["calls"]),
}))
"""
        try:
            value = json.loads(self._run(body, payload))
        except (TypeError, ValueError) as exc:
            raise Tau2Unavailable("official tau2 tool result was invalid") from exc
        self._state.update(
            {
                "tool_call_count": len(self._calls),
                "db_hash": value.get("db_hash"),
                "user_db_hash": value.get("user_db_hash"),
            }
        )
        return {
            "tool_message": value.get("tool_message"),
            "db_hash": value.get("db_hash"),
            "user_db_hash": value.get("user_db_hash"),
        }

    def restore_trajectory(self, trajectory: list[Mapping[str, Any]]) -> None:
        calls: list[dict[str, Any]] = []
        for record in trajectory:
            for raw in record.get("tool_calls") or ():
                calls.append(
                    {
                        "id": str(raw.get("call_id") or f"router-v2-{len(calls)}"),
                        "name": str(raw.get("name") or ""),
                        "arguments": dict(raw.get("arguments") or {}),
                        "requestor": "assistant",
                    }
                )
        self._calls = calls
        self._state["tool_call_count"] = len(calls)

    def state(self) -> Mapping[str, Any]:
        return dict(self._state)

    def evaluate(self, trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = {
            "task_set": self.task_set,
            "task_id": self.task_id,
            "calls": self._calls,
            "trajectory": {
                "turn_count": len(trajectory.get("trajectory") or [])
                if isinstance(trajectory, Mapping)
                else 0
            },
        }
        body = """
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.run import get_tasks

task = get_tasks(payload["task_set"], task_ids=[payload["task_id"]])[0]
domain = registry.get_env_constructor(payload["task_set"])()
initial = task.initial_state
if initial is not None:
    domain.set_state(initial.initialization_data, initial.initialization_actions, initial.message_history or [])
messages = []
for raw in payload["calls"]:
    call = ToolCall(**raw)
    tool_message = domain.get_response(call)
    messages.append(AssistantMessage.text("", tool_calls=[call]))
    messages.append(tool_message)
simulation = SimulationRun(
    id="router-v2-" + payload["task_id"],
    task_id=task.id,
    start_time="2026-01-01T00:00:00",
    end_time="2026-01-01T00:00:01",
    duration=1.0,
    termination_reason=TerminationReason.AGENT_STOP,
    messages=messages,
)
reward = evaluate_simulation(
    simulation, task, EvaluationType.ENV, False, payload["task_set"], strict_replay=False
)
print(json.dumps({
    "passed": bool(reward.reward == 1.0),
    "reward": reward.reward,
    "reward_info": reward.model_dump(mode="json"),
    "db_hash": domain.get_db_hash(),
    "user_db_hash": domain.get_user_db_hash(),
}))
"""
        try:
            value = json.loads(self._run(body, payload))
        except (TypeError, ValueError) as exc:
            raise Tau2Unavailable("official tau2 evaluation output was invalid") from exc
        return {
            "passed": bool(value.get("passed")),
            "reward": value.get("reward"),
            "reward_info": value.get("reward_info"),
            "db_hash": value.get("db_hash"),
            "user_db_hash": value.get("user_db_hash"),
            "official_evaluator_id": TAU2_EVALUATOR_ID,
            "official_evaluator_revision": TAU2_REVISION,
        }


def tau2_environment_factory(row: Any) -> PinnedTau2Environment:
    return PinnedTau2Environment(row)


__all__ = [
    "TAU2_EVALUATOR_ID",
    "TAU2_REVISION",
    "PinnedTau2Environment",
    "Tau2Unavailable",
    "tau2_available",
    "tau2_environment_factory",
]
