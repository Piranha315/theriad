"""Small worker-side service for disposable hosted agent containers.

This process is intended for separate worker capacity, never the web VPS. It
accepts an already-approved argv manifest, mounts only a task-owned workspace,
and applies Docker isolation flags before starting a container. It does not
accept shell strings, host paths, credentials, or arbitrary Docker arguments.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Theriad isolated agent runner", version="1.0")
_tasks: dict[str, dict[str, Any]] = {}


def _token() -> str:
    return os.environ.get("THERIAD_AGENT_RUNNER_TOKEN", "").strip()


def _authorize(authorization: str | None) -> None:
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    expected = _token()
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Runner authentication required.")


def _root() -> Path:
    root = Path(os.environ.get("THERIAD_AGENT_RUNNER_WORKDIR", "/var/lib/theriad-agent-tasks"))
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    return root


def _audit(event: str, record: dict[str, Any]) -> None:
    path = _root() / "audit.jsonl"
    safe = {
        "event": event,
        "at": time.time(),
        "task_id": record.get("task_id"),
        "organization_id": record.get("organization_id"),
        "user_id": record.get("user_id"),
        "status": record.get("status"),
        "container_id": record.get("container_id"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe, sort_keys=True) + "\n")


class StartRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    organization_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=120)
    argv: list[str] = Field(min_length=1, max_length=32)
    allowed_domains: list[str] = Field(default_factory=list, max_length=20)
    max_runtime_seconds: int = Field(default=300, ge=5, le=3600)
    memory_mb: int = Field(default=512, ge=128, le=4096)
    cpu_limit: float = Field(default=1.0, gt=0.1, le=4.0)
    approved: bool = False


class InputRequest(BaseModel):
    process_id: str = Field(min_length=1, max_length=120)
    text: str = Field(max_length=8000)


def _container_name(task_id: str) -> str:
    return f"theriad-agent-{task_id}"


def _workspace(task_id: str) -> Path:
    path = (_root() / task_id).resolve()
    if _root().resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid task workspace.")
    path.mkdir(mode=0o750, parents=True, exist_ok=True)
    return path


def _docker_command(body: StartRequest, workspace: Path) -> list[str]:
    image = os.environ.get("THERIAD_AGENT_RUNNER_IMAGE", "theriad-agent-runner:latest").strip()
    if not image or any(character.isspace() for character in image):
        raise HTTPException(status_code=503, detail="Runner image is not configured.")
    if body.allowed_domains:
        # A bare Docker network cannot safely enforce a domain allowlist. Do
        # not silently turn an allowlist request into unrestricted egress.
        raise HTTPException(status_code=409, detail="Network egress requires the configured worker proxy.")
    return [
        "docker", "run", "--detach", "--rm",
        "--name", _container_name(body.task_id),
        "--user", "65532:65532",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--network", "none",
        "--cpus", str(body.cpu_limit),
        "--memory", f"{body.memory_mb}m",
        "--pids-limit", "128",
        "--ulimit", "nproc=128:128",
        "--ulimit", "nofile=1024:1024",
        "--mount", f"type=bind,src={workspace},dst=/workspace,rw",
        "--workdir", "/workspace",
        image,
        *body.argv,
    ]


async def _enforce_runtime(task_id: str, container_id: str, seconds: int) -> None:
    """Stop a task when its worker-owned runtime budget expires."""
    await asyncio.sleep(seconds)
    record = _tasks.get(task_id)
    if not record or record.get("status") != "running":
        return
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "stop", "--time", "5", container_id,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
    except OSError:
        # The durable audit record still communicates that the budget expired;
        # the worker operator can clean up a dead Docker daemon separately.
        pass
    record["status"] = "timed_out"
    _audit("task_timed_out", record)


@app.get("/health")
def health() -> dict[str, Any]:
    docker_ready = bool(shutil.which("docker"))
    image_ready = bool(os.environ.get("THERIAD_AGENT_RUNNER_IMAGE", "theriad-agent-runner:latest").strip())
    configured = bool(_token())
    if not (docker_ready and image_ready and configured):
        raise HTTPException(status_code=503, detail="Runner prerequisites are not ready.")
    return {"status": "ok", "service": "theriad-agent-runner", "sandbox": "docker"}


@app.post("/v1/tasks")
async def start_task(body: StartRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    if not body.approved:
        raise HTTPException(status_code=409, detail="Task approval is required before container start.")
    if body.task_id in _tasks and _tasks[body.task_id].get("status") == "running":
        return _tasks[body.task_id]
    workspace = _workspace(body.task_id)
    command = _docker_command(body, workspace)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    except (OSError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Runner could not start the isolated container.") from exc
    if process.returncode != 0:
        raise HTTPException(status_code=503, detail="Runner rejected the isolated container request.")
    container_id = stdout.decode("utf-8", errors="replace").strip()[:128]
    record = {
        "task_id": body.task_id,
        "organization_id": body.organization_id,
        "user_id": body.user_id,
        "status": "running",
        "container_id": container_id,
        "workspace": "/workspace",
        "started_at": time.time(),
        "max_runtime_seconds": body.max_runtime_seconds,
    }
    _tasks[body.task_id] = record
    _audit("task_started", record)
    asyncio.create_task(_enforce_runtime(body.task_id, container_id, body.max_runtime_seconds))
    return record


@app.get("/v1/tasks/{task_id}")
def task_status(task_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    record = _tasks.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found.")
    return record


@app.post("/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    record = _tasks.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found.")
    if record.get("status") == "running":
        process = await asyncio.create_subprocess_exec(
            "docker", "stop", "--time", "5", record["container_id"],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
    record["status"] = "cancelled"
    _audit("task_cancelled", record)
    return record


@app.post("/v1/tasks/{task_id}/input")
def task_input(task_id: str, body: InputRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    raise HTTPException(status_code=409, detail="Interactive input requires a worker PTY adapter.")
