"""Resettable, caller-owned tool episodes with no generated-code execution.

The update worker never runs a model's shell commands. This evaluation harness
permits file tools inside a temporary fixture and reuses the existing strict
workspace-state oracle. Real Codex/Hermes executions remain separate pilots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .contracts import ChatRequest, GatewayError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def fixture_path(root: Path, relative: str):
    if not isinstance(relative, str):
        raise GatewayError("invalid_fixture_path", "Fixture path must be text", 409)
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise GatewayError("invalid_fixture_path", "File tool escaped the fixture", 409)
    return result


async def run_fixture(worker, deployment, task, protocol, policy):
    from theriad.routing.episode_runtime import grade_workspace_state
    from theriad.routing.episode_study import Episode

    fixture = task["fixture"]
    if not (fixture.get("expected_files") or fixture.get("expected_json")):
        raise GatewayError(
            "qualification_pending", "A deterministic final-state oracle is required", 409
        )
    max_calls = task.get("max_calls", 8)
    if type(max_calls) is not int or not 1 <= max_calls <= 20:
        raise GatewayError("invalid_suite", "Fixture episodes require 1 to 20 model calls", 409)
    episode = Episode(
        episode_id=task["id"],
        group_id=task["group"],
        family="tools",
        split="test" if task["split"] == "qualification" else task["split"],
        prompt="\n".join(str(m.get("content") or "") for m in task["request"]["messages"]),
        fixture=fixture,
        grader={"kind": "workspace"},
        source=task.get("source", "gateway-suite"),
    )
    body = {**task["request"], "tools": TOOLS}
    if (
        deployment.supported_parameters is None
        or "parallel_tool_calls" in deployment.supported_parameters
    ):
        body.setdefault("parallel_tool_calls", True)
    request = ChatRequest.model_validate(body)
    history = [m.model_dump(exclude_none=True) for m in request.messages]
    costs, outputs, latencies = [], [], []
    tool_time = 0.0
    started = time.monotonic()
    critical = True
    status = "step_limit"
    with TemporaryDirectory(prefix="theriad-gateway-eval-") as directory:
        root = Path(directory)
        for relative, content in fixture.get("files", {}).items():
            target = fixture_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for step in range(max_calls):
            part = {
                **task,
                "id": task["id"] + "/step/" + str(step),
                "grader": "protocol",
                "request": {**request.model_dump(exclude_none=True), "messages": history},
            }
            measurement = await worker.measure(deployment, part, protocol, policy)
            costs.append(measurement["cost"])
            outputs.append(measurement["output_tokens"])
            latencies.append(measurement["latency"])
            response = measurement["result"]
            choice = response["choices"][0]
            message = choice["message"]
            if choice["finish_reason"] in {"length", "content_filter"}:
                status = "truncated"
                break
            history.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
                }
            )
            if not message.get("tool_calls"):
                status = "completed"
                break
            for call in message["tool_calls"]:
                tool_started = time.monotonic()
                try:
                    args = json.loads(call["function"]["arguments"])
                    name = call["function"]["name"]
                    if name == "list_files" and not args:
                        result = {
                            "files": sorted(
                                str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                            )
                        }
                    elif name == "read_file" and set(args) == {"path"}:
                        result = {"content": fixture_path(root, args["path"]).read_text()}
                    elif (
                        name == "write_file"
                        and set(args) == {"path", "content"}
                        and isinstance(args["content"], str)
                    ):
                        if len(args["content"].encode()) > 1_000_000:
                            raise ValueError("file is too large")
                        target = fixture_path(root, args["path"])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(args["content"])
                        result = {"written": args["path"]}
                    else:
                        raise ValueError("unknown tool or arguments")
                except (ValueError, KeyError, OSError, GatewayError):
                    result = {"error": "invalid_or_unavailable_fixture_tool"}
                    critical = False
                tool_time += (time.monotonic() - tool_started) * 1000
                history.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                )
        success, evidence = grade_workspace_state(episode, root, status)
    return {
        "state": "completed",
        "success": float(bool(success)),
        "critical_pass": critical,
        "cost": sum(costs),
        "latency": sum(latencies) + tool_time,
        "output_tokens": sum(outputs),
        "protocol": protocol,
        "harness": "controlled-file-tools",
        "evidence": evidence,
        "calls": len(costs),
        "collection_wall_time_ms": (time.monotonic() - started) * 1000,
    }


def required_compatibility(deployment):
    cases = {"text", "streaming"}
    if deployment.tools:
        cases |= {
            "tool_only",
            "parallel_tools",
            "nullable_history",
            "stream_arguments",
            "tool_continuation",
        }
    if deployment.schemas:
        cases.add("json_schema")
    if deployment.native_responses:
        cases.add("native_responses")
    return cases
