"""Unified task, event, approval, and tool execution contract.

The engine is intentionally model-agnostic: planners submit typed tool calls,
while this layer owns permissions, durable events, cancellation, and adapters.
Existing chat/browser planners can therefore migrate without gaining a second
security policy implementation.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .process_manager import ProcessPolicyError, get_process_manager
from .project_tools import (
    ProjectToolError,
    delete_path,
    move_path,
    read_file,
    replace_text,
    search,
    write_file,
)
from .task_store import LocalTaskStore, TaskStore, build_task_store, supervisor_lease_seconds

PermissionProfile = Literal["strict", "autonomous"]
TaskState = Literal["ready", "running", "waiting_for_approval", "completed", "failed", "stopped"]
_SNAPSHOT_VERSION = 1

_WRITE_TOOLS = {
    "project.write", "project.replace", "project.patch", "project.move", "project.delete",
    "desktop.assign", "desktop.click", "desktop.type",
    "git.add", "git.commit",
}
_HARD_STOP_TOOLS = {"project.delete", "git.push", "external.send", "account.modify", "purchase"}
_SENSITIVE_FILENAMES = {".env", ".env.local", "credentials", "credentials.json", "secrets.json"}
_TERMINAL_TASK_STATES = frozenset({"completed", "failed", "stopped"})
_BROWSER_TOOLS = {
    "browser.opensandbox.start",
    "browser.opensandbox.observe",
    "browser.opensandbox.screenshot",
    "browser.opensandbox.navigate",
    "browser.opensandbox.action",
    "browser.opensandbox.stop",
}
_SENSITIVE_ARGUMENT_MARKERS = (
    "password",
    "passcode",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "private_key",
)


def _safe_text(value: Any, *, limit: int = 20_000) -> str:
    """Redact common credential-shaped values from durable/user-facing text."""
    text = str(value or "")[:limit]
    text = re.sub(
        r"(?i)(bearer\s+|(?:[?&](?:code|key|token|secret|password|passwd|auth|authorization)=))[^\s,;&]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(password|passcode|secret|token|api[_ -]?key|authorization|cookie|credential)\s*[:=]\s*([^\s,;]+)",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:sk|pk|ghp|github_pat|sb|bb_live)[-_][A-Za-z0-9._-]{8,}\b",
        "<redacted-key>",
        text,
    )
    return text


def _safe_arguments(value: Any, key: str = "") -> Any:
    """Redact credentials and typed form values before persistence or events."""
    normalized_key = key.lower().replace("-", "_")
    if any(marker in normalized_key for marker in _SENSITIVE_ARGUMENT_MARKERS):
        return "[REDACTED]"
    if normalized_key in {"screenshot", "screenshot_base64", "png_base64", "image", "image_base64"}:
        return "[REDACTED_IMAGE]"
    if normalized_key in {"text", "value", "input", "form_data", "fields", "body"}:
        return "[REDACTED_INPUT]" if value is not None else None
    if isinstance(value, dict):
        return {str(child_key): _safe_arguments(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_arguments(item) for item in value[:100]]
    if isinstance(value, str):
        return _safe_text(value)
    return value


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approved: bool = False


@dataclass
class ToolResult:
    call_id: str
    tool: str
    ok: bool
    output: Any = None
    error: str | None = None
    approval_required: bool = False


@dataclass
class ApprovalRequest:
    approval_id: str
    call: ToolCall
    reason: str
    created_at: float = field(default_factory=time.time)
    # Browser calls can contain a typed value.  Their durable snapshot keeps
    # only a redacted description, so a post-restart approval must be replayed
    # with a fresh call rather than executing the redacted snapshot.
    resubmission_required: bool = False


@dataclass
class TaskEvent:
    event_id: str
    task_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentTask:
    task_id: str
    goal: str
    workspace_root: str
    permission_profile: PermissionProfile
    requested_capabilities: list[str]
    allowed_domains: list[str]
    allowed_applications: list[str]
    budgets: dict[str, Any]
    owner_user_id: str | None = None
    owner_organization_id: str | None = None
    principal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: TaskState = "ready"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[TaskEvent] = field(default_factory=list, repr=False)
    pending_approval: ApprovalRequest | None = field(default=None, repr=False)
    stop_requested: bool = False
    context: list[dict[str, Any]] = field(default_factory=list, repr=False)
    final_answer: str | None = None
    checkpoint_count: int = 0

    def safe_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": _safe_text(self.goal),
            "workspace_root": self.workspace_root,
            "permission_profile": self.permission_profile,
            "requested_capabilities": self.requested_capabilities,
            "allowed_domains": self.allowed_domains,
            "allowed_applications": self.allowed_applications,
            "budgets": self.budgets,
            "metadata": _safe_arguments(self.metadata),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pending_approval": (
                {
                    "approval_id": self.pending_approval.approval_id,
                    "tool": self.pending_approval.call.name,
                    "reason": self.pending_approval.reason,
                }
                if self.pending_approval else None
            ),
            "event_count": len(self.events),
            "final_answer": self.final_answer,
            "checkpoint_count": self.checkpoint_count,
        }


class TaskEngineError(ValueError):
    pass


class TaskEngine:
    def __init__(
        self,
        event_dir: str | Path | None = None,
        allowed_workspace_root: str | Path | None = None,
        store: TaskStore | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.project_root = root
        self.allowed_workspace_root = Path(
            allowed_workspace_root
            or os.environ.get("THERIAD_ALLOWED_WORKSPACE_ROOT", str(root))
        ).expanduser().resolve()
        self.event_dir = Path(
            event_dir or os.environ.get("THERIAD_TASK_EVENT_DIR", str(root / "data" / "agent-tasks"))
        )
        self.store: TaskStore = store or (
            LocalTaskStore(self.event_dir)
            if event_dir is not None
            else build_task_store(self.event_dir)
        )
        self.supervisor_id = (
            os.environ.get("THERIAD_SUPERVISOR_ID")
            or f"{os.getpid()}-{uuid.uuid4()}"
        )
        self.lease_seconds = supervisor_lease_seconds()
        self._tasks: dict[str, AgentTask] = {}
        self._lock = threading.RLock()
        self._load_snapshots()

    def _snapshot_path(self, task_id: str) -> Path:
        return self.event_dir / f"{task_id}.snapshot.json"

    def _snapshot_payload(self, task: AgentTask) -> dict[str, Any]:
        pending = None
        if task.pending_approval is not None:
            call = task.pending_approval.call
            pending = {
                "approval_id": task.pending_approval.approval_id,
                # Never persist a pending browser action's typed value. A
                # post-restart approval must be resubmitted by the caller.
                "call": {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": _safe_arguments(call.arguments),
                    "approved": False,
                },
                "reason": task.pending_approval.reason,
                "created_at": task.pending_approval.created_at,
                "resubmission_required": task.pending_approval.resubmission_required,
            }
        return {
            "version": _SNAPSHOT_VERSION,
            "task": {
                "task_id": task.task_id,
                "goal": _safe_text(task.goal),
                "workspace_root": task.workspace_root,
                "permission_profile": task.permission_profile,
                "requested_capabilities": task.requested_capabilities,
                "allowed_domains": task.allowed_domains,
                "allowed_applications": task.allowed_applications,
                "budgets": task.budgets,
                "owner_user_id": task.owner_user_id,
                "owner_organization_id": task.owner_organization_id,
                "principal_id": task.principal_id,
                "metadata": _safe_arguments(task.metadata),
                "status": task.status,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "pending_approval": pending,
                "stop_requested": task.stop_requested,
                "context": _safe_arguments(task.context),
                "final_answer": task.final_answer,
                "checkpoint_count": task.checkpoint_count,
            },
        }

    def _write_snapshot(
        self,
        task: AgentTask,
        event: TaskEvent | None = None,
    ) -> None:
        task.checkpoint_count += 1
        self.store.persist(
            self._snapshot_payload(task),
            asdict(event) if event is not None else None,
        )

    def _load_snapshots(self) -> None:
        snapshots = self.store.load_snapshots()
        task_ids = [
            str(payload.get("task", {}).get("task_id") or "")
            for payload in snapshots
            if isinstance(payload, dict)
        ]
        events_by_task = self.store.load_events([item for item in task_ids if item])
        for payload in snapshots:
            task_id = str(payload.get("task", {}).get("task_id") or "")
            self._restore_snapshot(payload, events_by_task.get(task_id, []))

    def _restore_snapshot(
        self,
        payload: dict[str, Any],
        event_rows: list[dict[str, Any]],
    ) -> AgentTask | None:
        try:
            if payload.get("version") != _SNAPSHOT_VERSION:
                return None
            raw = payload["task"]
            workspace = Path(raw["workspace_root"]).expanduser().resolve()
            allowed = self.allowed_workspace_root
            if (
                not workspace.is_dir()
                or (workspace != allowed and allowed not in workspace.parents)
            ):
                return None
            pending_raw = raw.get("pending_approval")
            pending = None
            if pending_raw:
                pending_call = ToolCall(**pending_raw["call"])
                pending = ApprovalRequest(
                    approval_id=pending_raw["approval_id"],
                    call=pending_call,
                    reason=pending_raw["reason"],
                    created_at=float(pending_raw.get("created_at", time.time())),
                    # A browser approval snapshot is intentionally not
                    # executable: typed values and other sensitive arguments
                    # were redacted before persistence.
                    resubmission_required=bool(
                        pending_raw.get("resubmission_required")
                        or pending_call.name in _BROWSER_TOOLS
                    ),
                )
            status = raw.get("status", "ready")
            recovered_running = status == "running" and not self.store.status().get("shared")
            if recovered_running:
                status = "ready"
            task = AgentTask(
                task_id=raw["task_id"],
                goal=raw["goal"],
                workspace_root=str(workspace),
                permission_profile=raw["permission_profile"],
                requested_capabilities=list(raw.get("requested_capabilities", [])),
                allowed_domains=list(raw.get("allowed_domains", [])),
                allowed_applications=list(raw.get("allowed_applications", [])),
                budgets=dict(raw.get("budgets", {})),
                owner_user_id=raw.get("owner_user_id"),
                owner_organization_id=raw.get("owner_organization_id"),
                principal_id=raw.get("principal_id"),
                metadata=dict(raw.get("metadata", {})),
                status=status,
                created_at=float(raw.get("created_at", time.time())),
                updated_at=float(raw.get("updated_at", time.time())),
                pending_approval=pending,
                stop_requested=bool(raw.get("stop_requested", False)),
                context=list(raw.get("context", []))[-20:],
                final_answer=raw.get("final_answer"),
                checkpoint_count=int(raw.get("checkpoint_count", 0)),
            )
            for row in event_rows:
                try:
                    task.events.append(TaskEvent(**row))
                except (TypeError, ValueError):
                    continue
            self._tasks[task.task_id] = task
            if recovered_running:
                self.event(task, "task.recovered", {"previous_status": "running"})
            return task
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None

    def _workspace(self, value: str | None) -> Path:
        path = Path(value or self.project_root).expanduser().resolve()
        allowed = self.allowed_workspace_root
        if path != allowed and allowed not in path.parents:
            raise TaskEngineError(
                f"workspace_root must be inside configured root: {allowed}"
            )
        if not path.is_dir():
            raise TaskEngineError("workspace_root must be an existing directory")
        return path

    def create(
        self,
        *,
        goal: str,
        workspace_root: str | None = None,
        permission_profile: PermissionProfile = "strict",
        requested_capabilities: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        allowed_applications: list[str] | None = None,
        budgets: dict[str, Any] | None = None,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
        principal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTask:
        if permission_profile not in {"strict", "autonomous"}:
            raise TaskEngineError("permission_profile must be strict or autonomous")
        if not goal.strip():
            raise TaskEngineError("goal is required")
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            goal=goal.strip(),
            workspace_root=str(self._workspace(workspace_root)),
            permission_profile=permission_profile,
            requested_capabilities=sorted(set(requested_capabilities or [])),
            allowed_domains=sorted(set(allowed_domains or [])),
            allowed_applications=sorted(set(allowed_applications or [])),
            budgets=budgets or {},
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
            principal_id=principal_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._tasks[task.task_id] = task
        self.event(task, "task.created", task.safe_dict())
        return task

    def get(self, task_id: str) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            payload = self.store.load_snapshot(task_id)
            if payload is not None:
                events = self.store.load_events([task_id]).get(task_id, [])
                restored = self._restore_snapshot(payload, events)
                if restored is not None:
                    return restored
            raise TaskEngineError("Unknown task") from None

    def task_for_browser_session(self, session_id: str) -> AgentTask | None:
        """Find the canonical task that owns an OpenSandbox browser session."""
        candidate = str(session_id or "")
        if not candidate:
            return None
        with self._lock:
            for task in self._tasks.values():
                if task.metadata.get("browser_session_id") == candidate:
                    return task
        return None

    def event(self, task: AgentTask, kind: str, payload: dict[str, Any]) -> TaskEvent:
        with self._lock:
            item = TaskEvent(
                event_id=str(uuid.uuid4()),
                task_id=task.task_id,
                sequence=len(task.events) + 1,
                kind=kind,
                payload=payload,
            )
            task.events.append(item)
            task.updated_at = item.created_at
            self._write_snapshot(task, item)
            return item

    def checkpoint(self, task_id: str, *, metadata: dict[str, Any] | None = None) -> AgentTask:
        """Atomically persist resumable task state without adding a noisy event."""
        task = self.get(task_id)
        with self._lock:
            if metadata:
                task.metadata.update(metadata)
            task.updated_at = time.time()
            self._write_snapshot(task)
        return task

    def fail(self, task_id: str, error: str) -> AgentTask:
        task = self.get(task_id)
        task.status = "failed"
        task.metadata["error"] = str(error)[:1000]
        self.event(task, "task.failed", {"error": str(error)[:1000]})
        self._schedule_browser_cleanup(task_id)
        return task

    async def fail_async(self, task_id: str, error: str) -> AgentTask:
        """Fail a task and await release of its OpenSandbox browser, if any."""
        task = self.get(task_id)
        task.status = "failed"
        task.metadata["error"] = str(error)[:1000]
        self.event(task, "task.failed", {"error": str(error)[:1000]})
        await self.cleanup_browser_session(task_id)
        return task

    def events(self, task_id: str, *, after: int = 0) -> list[dict]:
        return [asdict(item) for item in self.get(task_id).events if item.sequence > after]

    def storage_status(self) -> dict[str, Any]:
        return {
            **self.store.status(),
            "supervisor_id": self.supervisor_id,
            "lease_seconds": self.lease_seconds,
        }

    def heartbeat(self, task_id: str) -> bool:
        return self.store.heartbeat(task_id, self.supervisor_id, self.lease_seconds)

    async def supervise(self, task_id: str) -> dict[str, Any]:
        """Claim and drive one task through the canonical planner loop."""
        task = self.get(task_id)
        if task.status in {"completed", "failed", "stopped"}:
            return task.safe_dict()
        if not self.store.claim(task_id, self.supervisor_id, self.lease_seconds):
            raise TaskEngineError("Task is already claimed by another supervisor.")
        if task.status == "running":
            task.status = "ready"
            self.event(task, "task.recovered", {"previous_status": "running"})
        self.event(task, "supervisor.claimed", {"supervisor_id": self.supervisor_id})
        try:
            from .task_runner import run_task

            return await run_task(task_id, engine=self, supervised=True)
        finally:
            try:
                self.event(task, "supervisor.released", {"supervisor_id": self.supervisor_id})
            finally:
                self.store.release(task_id, self.supervisor_id)

    def attach_specialist(
        self,
        task_id: str,
        *,
        kind: str,
        specialist_id: str,
        details: dict[str, Any] | None = None,
    ) -> AgentTask:
        """Register a browser/terminal specialist under this supervisor."""
        task = self.get(task_id)
        specialists = task.metadata.setdefault("specialists", {})
        specialists[kind] = {
            "id": specialist_id,
            "status": "running",
            **(details or {}),
        }
        task.status = "running"
        self.event(task, "specialist.attached", {
            "kind": kind,
            "specialist_id": specialist_id,
            **(details or {}),
        })
        return task

    def sync_specialist(
        self,
        task_id: str,
        *,
        kind: str,
        status: str,
        final_answer: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AgentTask:
        """Project specialist state into the canonical task lifecycle."""
        task = self.get(task_id)
        specialists = task.metadata.setdefault("specialists", {})
        prior = dict(specialists.get(kind, {}))
        current = {
            **prior,
            "status": status,
            "final_answer": final_answer,
            "error": error,
            **(details or {}),
        }
        specialists[kind] = current
        states = {str(item.get("status")) for item in specialists.values()}
        if "failed" in states:
            failed = next(
                (item for item in specialists.values() if item.get("status") == "failed"),
                current,
            )
            if task.status != "failed":
                return self.fail(task_id, failed.get("error") or f"{kind} specialist failed")
            return task
        if states and states <= {"completed"}:
            answers = [
                str(item.get("final_answer"))
                for item in specialists.values()
                if item.get("final_answer")
            ]
            if task.status != "completed":
                self.execute(
                    task_id,
                    ToolCall("task.complete", {"answer": final_answer or "\n\n".join(answers) or "Task completed."}),
                )
            return task
        if states.intersection({"stopped", "stopped_budget_exceeded"}):
            task.status = "stopped"
        elif states.intersection({"waiting_for_approval", "needs_approval", "waiting_for_input", "manual_login_pause"}):
            task.status = "waiting_for_approval"
        else:
            task.status = "running"
        if prior.get("status") != status or details:
            self.event(task, "specialist.status", {
                "kind": kind,
                "status": status,
                "error": error,
                **(details or {}),
            })
        else:
            self.checkpoint(task_id)
        return task

    def stop(self, task_id: str) -> AgentTask:
        task = self.get(task_id)
        task.stop_requested = True
        task.status = "stopped"
        get_process_manager().cleanup_task(task_id)
        self.event(task, "task.stopped", {})
        self._schedule_browser_cleanup(task_id)
        return task

    async def stop_async(self, task_id: str) -> AgentTask:
        """Stop a task and await release of its OpenSandbox browser session."""
        task = self.get(task_id)
        task.stop_requested = True
        task.status = "stopped"
        get_process_manager().cleanup_task(task_id)
        self.event(task, "task.stopped", {})
        await self.cleanup_browser_session(task_id)
        return task

    def _schedule_browser_cleanup(self, task_id: str) -> None:
        """Release a browser from sync callers without crossing event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.cleanup_browser_session(task_id))
            except Exception:
                # Terminal state must remain durable even if cleanup is
                # unavailable during interpreter shutdown.
                pass
        else:
            loop.create_task(self.cleanup_browser_session(task_id))

    async def cleanup_browser_session(self, task_id: str) -> dict[str, Any] | None:
        """Best-effort, owner-scoped cleanup for a task-owned OpenSandbox."""
        task = self.get(task_id)
        session_id = str(task.metadata.get("browser_session_id") or "").strip()
        if not session_id:
            return None
        from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

        try:
            result = await get_opensandbox_browser_runtime().stop_session(
                session_id,
                owner_user_id=task.owner_user_id,
                owner_organization_id=task.owner_organization_id,
            )
        except Exception as exc:
            # Preserve the original task result, but leave an operator-visible
            # safe marker so a leaked remote session is diagnosable.
            task.metadata["browser_cleanup_error"] = _safe_text(exc, limit=500)
            self.checkpoint(task_id)
            return {"ok": False, "status": "cleanup_failed", "error": task.metadata["browser_cleanup_error"]}

        task.metadata["browser_session_status"] = result.get("status")
        self.checkpoint(task_id)
        return result

    def _approval(self, task: AgentTask, call: ToolCall, reason: str) -> ToolResult:
        request = ApprovalRequest(
            str(uuid.uuid4()),
            call,
            reason,
            resubmission_required=False,
        )
        task.pending_approval = request
        task.status = "waiting_for_approval"
        self.event(task, "approval.required", {
            "approval_id": request.approval_id,
            "call_id": call.call_id,
            "tool": call.name,
            "reason": reason,
        })
        return ToolResult(call.call_id, call.name, False, error=reason, approval_required=True)

    def approve(self, task_id: str, approval_id: str, *, approved: bool) -> ToolResult:
        task = self.get(task_id)
        pending = task.pending_approval
        if pending is None or pending.approval_id != approval_id:
            raise TaskEngineError("Unknown or stale approval")
        if not approved:
            task.pending_approval = None
            task.status = "running"
            self.event(task, "approval.denied", {"approval_id": approval_id})
            return ToolResult(pending.call.call_id, pending.call.name, False, error="User denied approval")
        if pending.resubmission_required:
            # The restored snapshot contains only redacted arguments. Keep the
            # approval pending and durable until a fresh replacement call is
            # supplied; never execute the redacted placeholder values.
            task.status = "waiting_for_approval"
            self.event(task, "approval.resubmission_required", {
                "approval_id": approval_id,
                "tool": pending.call.name,
            })
            return ToolResult(
                pending.call.call_id,
                pending.call.name,
                False,
                error=(
                    "The browser approval survived a restart without its "
                    "sensitive arguments. Resubmit the canonical browser tool "
                    "call with its values after approval."
                ),
                approval_required=True,
            )
        task.pending_approval = None
        self.event(task, "approval.granted", {"approval_id": approval_id})
        pending.call.approved = True
        task.status = "running"
        return self.execute(task_id, pending.call)

    async def approve_async(
        self,
        task_id: str,
        approval_id: str,
        *,
        approved: bool,
        replacement_call: ToolCall | None = None,
    ) -> ToolResult:
        """Resolve approval without moving OpenSandbox handles between loops."""
        task = self.get(task_id)
        pending = task.pending_approval
        if pending is None or pending.approval_id != approval_id:
            raise TaskEngineError("Unknown or stale approval")
        if not approved:
            task.pending_approval = None
            task.status = "running"
            self.event(task, "approval.denied", {"approval_id": approval_id})
            return ToolResult(
                pending.call.call_id,
                pending.call.name,
                False,
                error="User denied approval",
            )
        if pending.resubmission_required:
            if replacement_call is None:
                task.status = "waiting_for_approval"
                task.pending_approval = pending
                self.event(task, "approval.resubmission_required", {
                    "approval_id": approval_id,
                    "tool": pending.call.name,
                })
                return ToolResult(
                    pending.call.call_id,
                    pending.call.name,
                    False,
                    error=(
                        "The browser approval survived a restart without its "
                        "sensitive arguments. Resubmit the canonical browser "
                        "tool call with its values."
                    ),
                    approval_required=True,
                )
            if replacement_call.name != pending.call.name:
                raise TaskEngineError("Replacement call does not match the pending browser tool")
            if replacement_call.call_id == pending.call.call_id:
                raise TaskEngineError("Replacement call must use a fresh call_id")
            task.pending_approval = None
            self.event(task, "approval.granted", {"approval_id": approval_id})
            replacement_call.approved = True
            return await self.execute_async(task_id, replacement_call)
        task.pending_approval = None
        self.event(task, "approval.granted", {"approval_id": approval_id})
        pending.call.approved = True
        task.status = "running"
        return await self.execute_async(task_id, pending.call)

    def execute(self, task_id: str, call: ToolCall) -> ToolResult:
        task = self.get(task_id)
        guard = self._validate_call(task, call)
        if guard is not None:
            return guard
        if call.name in _BROWSER_TOOLS:
            # Playwright objects are loop-bound.  Refuse the synchronous API
            # instead of launching a second browser path or crossing loops.
            return self._failed_result(
                task,
                call,
                "OpenSandbox browser tools require execute_async().",
            )

        task.status = "running"
        self.event(task, "tool.started", {"call_id": call.call_id, "tool": call.name})
        try:
            output = self._dispatch(task, call)
            return self._completed_result(task, call, output)
        except ProcessPolicyError as exc:
            if exc.approval_required and not call.approved:
                return self._approval(task, call, str(exc))
            return self._failed_result(task, call, str(exc))
        except (ProjectToolError, TaskEngineError, KeyError, OSError, ValueError) as exc:
            return self._failed_result(task, call, str(exc))
        except Exception as exc:
            return self._failed_result(task, call, f"{type(exc).__name__}: {exc}")

    async def execute_async(self, task_id: str, call: ToolCall) -> ToolResult:
        """Execute a task call without moving OpenSandbox Playwright objects across loops."""
        task = self.get(task_id)
        guard = self._validate_call(task, call)
        if guard is not None:
            return guard

        task.status = "running"
        self.event(task, "tool.started", {"call_id": call.call_id, "tool": call.name})
        try:
            if call.name in _BROWSER_TOOLS:
                output = await self._dispatch_opensandbox(task, call)
            else:
                output = await asyncio.to_thread(self._dispatch, task, call)
            if (
                call.name in _BROWSER_TOOLS
                and isinstance(output, dict)
                and output.get("status") == "waiting_for_approval"
            ):
                return self._waiting_result(task, call, output)
            return self._completed_result(task, call, output)
        except ProcessPolicyError as exc:
            if exc.approval_required and not call.approved:
                return self._approval(task, call, str(exc))
            return self._failed_result(task, call, str(exc))
        except (ProjectToolError, TaskEngineError, KeyError, OSError, ValueError) as exc:
            result = self._failed_result(task, call, str(exc))
            if call.name in _BROWSER_TOOLS:
                await self.fail_async(task_id, str(exc))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = self._failed_result(task, call, error)
            if call.name in _BROWSER_TOOLS:
                await self.fail_async(task_id, error)
            return result

    def _validate_call(self, task: AgentTask, call: ToolCall) -> ToolResult | None:
        if task.stop_requested:
            return ToolResult(call.call_id, call.name, False, error="Task is stopped")
        if task.pending_approval:
            if not call.approved:
                return ToolResult(call.call_id, call.name, False, error="Another approval is pending")
            if call.call_id != task.pending_approval.call.call_id:
                return ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error="Resolve the existing approval before submitting another approved call.",
                )
        if call.name.startswith("browser."):
            if call.name not in _BROWSER_TOOLS:
                return ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=(
                        f"{call.name} is deprecated. Use browser.opensandbox.* tools."
                    ),
                )
            if call.name in {"browser.opensandbox.navigate", "browser.opensandbox.action"}:
                from theriad.browser.policy import BrowserPolicy

                logical_action = self._browser_action(call)
                decision = BrowserPolicy().classify(logical_action, approved=call.approved)
                if not decision.allowed:
                    if decision.status == "approval_required" and not call.approved:
                        return self._approval(task, call, decision.reason)
                    return ToolResult(call.call_id, call.name, False, error=decision.reason)
        if call.name in _HARD_STOP_TOOLS and not call.approved:
            return self._approval(task, call, f"{call.name} has irreversible or external effects.")
        if call.name == "project.read" and not call.approved:
            filename = Path(str(call.arguments.get("path", ""))).name.lower()
            if filename in _SENSITIVE_FILENAMES or filename.endswith((".pem", ".key", ".p12")):
                return self._approval(task, call, "Reading a credential-bearing file requires explicit approval.")
        if call.name in _WRITE_TOOLS and task.permission_profile == "strict" and not call.approved:
            return self._approval(task, call, f"{call.name} changes workspace state in strict mode.")
        if call.name == "terminal.run" and not call.approved:
            from .terminal_tool import evaluate_command, get_terminal_mode

            command = str(call.arguments.get("command") or "")
            decision = evaluate_command(command, project_root=task.workspace_root)
            if decision.verdict == "needs_approval" and get_terminal_mode() == "safe":
                return self._approval(task, call, decision.reason)
        return None

    def _completed_result(self, task: AgentTask, call: ToolCall, output: Any) -> ToolResult:
        result = ToolResult(call.call_id, call.name, True, output=output)
        task.context.append(self._context_record(call, output))
        task.context[:] = task.context[-20:]
        self.event(task, "tool.completed", {
            "call_id": call.call_id,
            "tool": call.name,
            "output_type": type(output).__name__,
        })
        return result

    def _failed_result(self, task: AgentTask, call: ToolCall, error: str) -> ToolResult:
        safe_error = str(_safe_arguments(error))
        result = ToolResult(call.call_id, call.name, False, error=safe_error)
        self.event(task, "tool.failed", {
            "call_id": call.call_id,
            "tool": call.name,
            "error": safe_error,
        })
        return result

    def _waiting_result(
        self,
        task: AgentTask,
        call: ToolCall,
        output: dict[str, Any],
    ) -> ToolResult:
        """Represent a runtime pause without falsely completing the tool."""
        task.status = "waiting_for_approval"
        task.context.append(self._context_record(call, output))
        task.context[:] = task.context[-20:]
        self.event(task, "tool.waiting", {
            "call_id": call.call_id,
            "tool": call.name,
            "status": output.get("status"),
            "captcha_detected": bool(output.get("captcha_detected")),
            "pending_action": output.get("pending_action"),
        })
        return ToolResult(
            call.call_id,
            call.name,
            False,
            output=output,
            error=output.get("error") or "Browser runtime is waiting for approval.",
            approval_required=True,
        )

    @staticmethod
    def _browser_action(call: ToolCall) -> str:
        explicit = str(
            call.arguments.get("action_type")
            or call.arguments.get("action")
            or call.arguments.get("type")
            or ""
        ).strip().lower()
        if isinstance(call.arguments.get("action"), dict):
            action_payload = call.arguments["action"]
            explicit = str(
                action_payload.get("action_type")
                or action_payload.get("action")
                or action_payload.get("type")
                or ""
            ).strip().lower()
        if explicit:
            return explicit
        target = " ".join(
            str(call.arguments.get(key) or "")
            for key in ("target", "selector", "text", "url")
        ).lower()
        risky = {
            "purchase": ("purchase", "buy now", "place order"),
            "payment": ("pay", "payment"),
            "checkout": ("checkout",),
            "send_message": ("send", "message", "email"),
            "delete": ("delete", "remove account"),
            "login": ("login", "sign in"),
            "submit_form": ("submit", "apply now"),
        }
        for action, markers in risky.items():
            if any(marker in target for marker in markers):
                return action
        return call.name.rsplit(".", 1)[-1]

    @staticmethod
    def _context_record(call: ToolCall, output: Any) -> dict[str, Any]:
        secret_keys = {"password", "token", "authorization", "cookie", "api_key", "secret"}

        def clean(value: Any, key: str = "") -> Any:
            low = key.lower()
            if any(marker in low for marker in secret_keys):
                return "[REDACTED]"
            if low in {"png_base64", "screenshot", "screenshot_base64"}:
                return "[IMAGE AVAILABLE TO VISION ADAPTER]"
            if isinstance(value, dict):
                return {str(k): clean(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value[:100]]
            if isinstance(value, str):
                return value[:20_000]
            return value

        arguments = _safe_arguments(call.arguments)
        filename = Path(str(call.arguments.get("path", ""))).name.lower()
        if call.name == "project.read" and (
            filename in _SENSITIVE_FILENAMES or filename.endswith((".pem", ".key", ".p12"))
        ):
            output = {"path": call.arguments.get("path"), "content": "[REDACTED_CREDENTIAL_FILE]"}
        return {"tool": call.name, "arguments": arguments, "result": _safe_arguments(clean(output))}

    async def _dispatch_opensandbox(self, task: AgentTask, call: ToolCall) -> Any:
        """Dispatch the canonical browser tools to one task-owned sandbox."""
        from theriad.browser.opensandbox_runtime import (
            get_opensandbox_browser_runtime,
        )

        runtime = get_opensandbox_browser_runtime()
        args = dict(call.arguments or {})
        owner = {
            "owner_user_id": task.owner_user_id,
            "owner_organization_id": task.owner_organization_id,
        }
        session_id = task.metadata.get("browser_session_id")

        if call.name == "browser.opensandbox.start":
            requested_session_id = str(args.get("session_id") or "").strip() or None
            if session_id and requested_session_id and requested_session_id != session_id:
                raise TaskEngineError("A task can use only its own OpenSandbox browser session.")
            viewport = args.get("viewport")
            if isinstance(viewport, dict):
                viewport = {
                    "width": max(320, min(int(viewport.get("width", 1280)), 3840)),
                    "height": max(240, min(int(viewport.get("height", 900)), 2160)),
                }
            else:
                viewport = None
            result = await runtime.start_session(
                session_id=session_id or requested_session_id,
                start_url=args.get("start_url") or args.get("url"),
                user_prompt=str(args.get("user_prompt") or task.goal),
                visible=bool(args.get("visible", False)),
                viewport=viewport,
                allowed_domains=task.allowed_domains,
                **owner,
            )
            returned_session_id = result.get("session_id")
            if returned_session_id:
                task.metadata["browser_session_id"] = str(returned_session_id)
                task.metadata["browser_backend"] = "opensandbox"
                self.checkpoint(task.task_id)
        else:
            if not session_id:
                raise TaskEngineError(
                    "Start browser.opensandbox.start before using the browser session."
                )
            if call.name == "browser.opensandbox.observe":
                result = await runtime.observe(
                    str(session_id),
                    screenshot=bool(args.get("screenshot", False)),
                    **owner,
                )
            elif call.name == "browser.opensandbox.screenshot":
                result = await runtime.observe(str(session_id), screenshot=True, **owner)
            elif call.name == "browser.opensandbox.navigate":
                url = str(args.get("url") or "").strip()
                if not url:
                    raise TaskEngineError("browser.opensandbox.navigate requires url")
                result = await runtime.navigate(str(session_id), url, **owner)
            elif call.name == "browser.opensandbox.action":
                raw_action = args.get("action")
                action = dict(raw_action) if isinstance(raw_action, dict) else dict(args)
                if not action.get("action"):
                    action["action"] = args.get("action_type") or args.get("type")
                action.pop("session_id", None)
                action.pop("approved", None)
                if not action.get("action"):
                    raise TaskEngineError(
                        "browser.opensandbox.action requires an action name"
                    )
                result = await runtime.action(
                    str(session_id),
                    action,
                    approved=call.approved,
                    **owner,
                )
            elif call.name == "browser.opensandbox.stop":
                result = await runtime.stop_session(str(session_id), **owner)
            else:  # Defensive: _validate_call owns the public allowlist.
                raise TaskEngineError(f"Unsupported browser tool: {call.name}")

        if not isinstance(result, dict):
            raise TaskEngineError("OpenSandbox returned an invalid browser result.")
        status = str(result.get("status") or "failed")
        if status == "waiting_for_approval":
            task.status = "waiting_for_approval"
        elif status == "stopped":
            task.status = "stopped"
        if not result.get("ok", False) and status not in {"waiting_for_approval"}:
            detail = result.get("error") or "OpenSandbox browser operation failed."
            setup = result.get("setup")
            if setup:
                detail = f"{detail} {setup}"
            raise TaskEngineError(str(detail))
        return result

    def _dispatch(self, task: AgentTask, call: ToolCall) -> Any:
        args = call.arguments
        root = task.workspace_root
        if call.name == "web.search":
            import asyncio
            from urllib.parse import urlsplit

            from theriad.search.service import SearchService

            query = str(args.get("query") or "").strip()
            if not query:
                raise TaskEngineError("web.search requires query")
            limit = max(1, min(int(args.get("limit", 8)), 8))
            rows = asyncio.run(SearchService().search(query))
            if task.allowed_domains:
                allowed = {item.lower().lstrip(".") for item in task.allowed_domains}
                rows = [
                    row for row in rows
                    if any(
                        (urlsplit(row.url).hostname or "").lower() == domain
                        or (urlsplit(row.url).hostname or "").lower().endswith("." + domain)
                        for domain in allowed
                    )
                ]
            rows = rows[:limit]
            return {
                "query": query,
                "provider": rows[0].source if rows else None,
                "results": [row.to_dict() for row in rows],
                "number_of_results": len(rows),
            }
        if call.name == "web.fetch":
            import asyncio
            from urllib.parse import urlsplit

            from theriad.search.evidence import fetch_page_evidence

            url = str(args.get("url") or "").strip()
            if not url:
                raise TaskEngineError("web.fetch requires url")
            host = (urlsplit(url).hostname or "").lower()
            if task.allowed_domains:
                allowed = {item.lower().lstrip(".") for item in task.allowed_domains}
                if not any(host == domain or host.endswith("." + domain) for domain in allowed):
                    raise TaskEngineError("Fetch domain is not in this task's allowlist.")
            result = asyncio.run(fetch_page_evidence(url))
            if result.get("status") == "blocked_url":
                raise TaskEngineError("web.fetch only permits public HTTP(S) URLs")
            return result
        if call.name == "project.read":
            return read_file(root, args["path"])
        if call.name == "project.search":
            return search(root, args["query"], limit=int(args.get("limit", 100)))
        if call.name == "project.write":
            return write_file(root, args["path"], args.get("content", ""), overwrite=bool(args.get("overwrite")))
        if call.name in {"project.replace", "project.patch"}:
            return replace_text(root, args["path"], args["expected"], args.get("replacement", ""), count=int(args.get("count", 1)))
        if call.name == "project.move":
            return move_path(root, args["source"], args["destination"])
        if call.name == "project.delete":
            return delete_path(root, args["path"])
        if call.name == "terminal.start":
            return get_process_manager().start(
                task_id=task.task_id,
                command=args["command"],
                workspace_root=root,
                cwd=args.get("cwd"),
                permission_profile=task.permission_profile,
                approved=call.approved,
            )
        if call.name == "terminal.run":
            from .terminal_tool import TerminalTool

            command = str(args.get("command") or "").strip()
            if not command:
                raise TaskEngineError("terminal.run requires command")
            result = TerminalTool(project_root=root).run(command, approved=call.approved)
            if result.verdict == "blocked":
                raise TaskEngineError(result.reason)
            return result.to_dict()
        if call.name == "terminal.read":
            return get_process_manager().read(args["process_id"])
        if call.name == "terminal.write":
            return get_process_manager().write(args["process_id"], args.get("text", ""))
        if call.name == "terminal.stop":
            return get_process_manager().stop(args["process_id"])
        if call.name.startswith("git."):
            action = call.name.removeprefix("git.")
            if action == "status":
                command = "git status --short --branch"
            elif action == "diff":
                command = "git diff" + (" --stat" if args.get("stat") else "")
            elif action == "log":
                command = f"git log -{max(1, min(int(args.get('count', 10)), 100))} --oneline --decorate"
            elif action == "branch":
                command = "git branch --show-current"
            elif action == "add":
                paths = args.get("paths") or []
                if not paths:
                    raise TaskEngineError("git.add requires paths")
                command = "git add -- " + " ".join(shlex.quote(str(path)) for path in paths)
            elif action == "commit":
                command = "git commit -m " + shlex.quote(str(args.get("message") or ""))
            elif action == "push":
                command = "git push"
                if args.get("remote"):
                    command += " " + shlex.quote(str(args["remote"]))
                if args.get("branch"):
                    command += " " + shlex.quote(str(args["branch"]))
            else:
                raise TaskEngineError(f"Unsupported git tool: {action}")
            return get_process_manager().start(
                task_id=task.task_id,
                command=command,
                workspace_root=root,
                permission_profile=task.permission_profile,
                approved=call.approved,
            )
        if call.name.startswith(("browser.extension.", "browser.devtools.")):
            raise TaskEngineError(
                f"{call.name} is deprecated. Use browser.opensandbox.* tools."
            )
        if call.name.startswith("desktop."):
            from .desktop_control import get_computer_use
            computer = get_computer_use()
            action = call.name.removeprefix("desktop.")
            if action == "assign":
                return computer.assign(task.task_id, args["application"], task.allowed_applications)
            if action == "screenshot":
                return computer.screenshot(task.task_id)
            if action == "click":
                return computer.click(task.task_id, float(args["x"]), float(args["y"]))
            if action == "type":
                return computer.type_text(task.task_id, str(args.get("text", "")))
            if action == "stop":
                return computer.stop(task.task_id)
        if call.name == "task.complete":
            task.final_answer = str(args.get("answer") or "Task completed.")
            task.status = "completed"
            return {"completed": True, "answer": task.final_answer}
        raise TaskEngineError(f"Unknown tool: {call.name}")


_ENGINE: TaskEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_task_engine() -> TaskEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = TaskEngine()
    return _ENGINE
