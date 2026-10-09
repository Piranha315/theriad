"""Guarded host process lifecycle for local agent tasks.

This is deliberately *not* a sandbox.  It provides deterministic policy,
workspace scoping, a minimal environment, PTY streaming, process-group
termination, and audit-friendly results while executing on the host.
"""
from __future__ import annotations

import os
import pty
import select
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .terminal_tool import evaluate_command, redact_secrets

PermissionProfile = Literal["strict", "autonomous"]

_ALWAYS_APPROVAL_PREFIXES = (
    ("git", "push"),
    ("git", "commit"),
    ("git", "tag"),
    ("npm", "publish"),
    ("python", "-m", "twine"),
    ("twine",),
    ("gh", "pr", "create"),
    ("gh", "release"),
)

_NETWORK_EXECUTABLES = {"curl", "wget", "nc", "ncat", "scp", "rsync", "ssh"}
_HOST_DANGEROUS_EXECUTABLES = {"sudo", "su", "security", "launchctl", "crontab", "dscl"}
_SHELL_EXECUTABLES = {"sh", "bash", "zsh", "fish", "osascript"}
_AUTONOMOUS_EXECUTABLES = {
    "python", "python3", "pytest", "npm", "npx", "node", "git", "ruff", "mypy",
    "cargo", "rustc", "go", "make", "cmake", "pip", "pip3", "uv", "pnpm", "yarn",
    "pwd", "ls", "find", "rg", "grep", "cat", "head", "tail", "wc",
}
_OUTPUT_LIMIT = 200_000


@dataclass
class ProcessRecord:
    process_id: str
    task_id: str
    command: str
    argv: list[str]
    cwd: str
    permission_profile: PermissionProfile
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    exit_code: int | None = None
    output: deque[str] = field(default_factory=deque, repr=False)
    output_chars: int = 0
    read_offset: int = 0
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    master_fd: int | None = field(default=None, repr=False)
    pump_thread: threading.Thread | None = field(default=None, repr=False)

    def snapshot(self, *, include_output: bool = False) -> dict:
        data = {
            "process_id": self.process_id,
            "task_id": self.task_id,
            "command": self.command,
            "argv": self.argv,
            "cwd": self.cwd,
            "permission_profile": self.permission_profile,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "guarded_host_execution": True,
            "sandboxed": False,
        }
        if include_output:
            data["output"] = redact_secrets("".join(self.output))
        return data


class ProcessPolicyError(ValueError):
    def __init__(self, message: str, *, approval_required: bool = False) -> None:
        super().__init__(message)
        self.approval_required = approval_required


class HostProcessManager:
    def __init__(self) -> None:
        self._records: dict[str, ProcessRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _inside(path: str | Path, root: str | Path) -> bool:
        resolved = Path(path).expanduser().resolve()
        base = Path(root).expanduser().resolve()
        return resolved == base or base in resolved.parents

    def authorize(
        self,
        command: str,
        *,
        workspace_root: str,
        permission_profile: PermissionProfile,
        approved: bool = False,
    ) -> list[str]:
        decision = evaluate_command(command, project_root=workspace_root)
        if decision.verdict == "blocked":
            raise ProcessPolicyError(decision.reason)
        argv = decision.argv or shlex.split(command)
        executable = Path(argv[0]).name if argv else ""
        if executable in _HOST_DANGEROUS_EXECUTABLES:
            raise ProcessPolicyError(f"{executable} is blocked in guarded host mode.")
        prefix = tuple(argv)
        irreversible = any(prefix[: len(item)] == item for item in _ALWAYS_APPROVAL_PREFIXES)
        networked = executable in _NETWORK_EXECUTABLES
        if irreversible and not approved:
            raise ProcessPolicyError(
                "Publishing, pushing, committing, or creating external releases requires explicit approval.",
                approval_required=True,
            )
        if networked and not approved:
            raise ProcessPolicyError(
                "Direct network-transfer commands require explicit approval in guarded host mode.",
                approval_required=True,
            )
        if executable in _SHELL_EXECUTABLES and not approved:
            raise ProcessPolicyError(
                f"{executable} can bypass argv-level workspace checks and requires explicit approval.",
                approval_required=True,
            )
        if executable in {"python", "python3", "node"} and any(arg in {"-c", "-e", "--eval"} for arg in argv[1:]) and not approved:
            raise ProcessPolicyError(
                "Inline interpreter code requires explicit approval in guarded host mode.",
                approval_required=True,
            )
        if decision.verdict == "needs_approval" and permission_profile == "strict" and not approved:
            raise ProcessPolicyError(decision.reason, approval_required=True)
        if decision.verdict == "needs_approval" and permission_profile == "autonomous" and executable not in _AUTONOMOUS_EXECUTABLES and not approved:
            raise ProcessPolicyError(
                f"'{executable}' is outside the autonomous development-tool set and requires approval.",
                approval_required=True,
            )
        return argv

    def start(
        self,
        *,
        task_id: str,
        command: str,
        workspace_root: str,
        cwd: str | None = None,
        permission_profile: PermissionProfile = "strict",
        approved: bool = False,
    ) -> dict:
        root = Path(workspace_root).expanduser().resolve()
        workdir = Path(cwd or root).expanduser().resolve()
        if not self._inside(workdir, root):
            raise ProcessPolicyError("Working directory must stay inside the selected workspace.")
        argv = self.authorize(
            command,
            workspace_root=str(root),
            permission_profile=permission_profile,
            approved=approved,
        )
        master_fd, slave_fd = pty.openpty()
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL"}
        }
        env.update({"HOME": str(root), "THERIAD_TASK_ID": task_id})
        proc = subprocess.Popen(
            argv,
            cwd=str(workdir),
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        record = ProcessRecord(
            process_id=str(uuid.uuid4()),
            task_id=task_id,
            command=command,
            argv=argv,
            cwd=str(workdir),
            permission_profile=permission_profile,
            proc=proc,
            master_fd=master_fd,
        )
        with self._lock:
            self._records[record.process_id] = record
        record.pump_thread = threading.Thread(target=self._pump, args=(record,), daemon=True)
        record.pump_thread.start()
        return record.snapshot()

    def _pump(self, record: ProcessRecord) -> None:
        assert record.proc is not None and record.master_fd is not None
        try:
            while record.proc.poll() is None:
                readable, _, _ = select.select([record.master_fd], [], [], 0.25)
                if readable:
                    self._append(record, os.read(record.master_fd, 8192))
            while True:
                readable, _, _ = select.select([record.master_fd], [], [], 0)
                if not readable:
                    break
                self._append(record, os.read(record.master_fd, 8192))
        except (OSError, EOFError):
            pass
        finally:
            # A PTY can report EOF/EIO just before the child process has been
            # reaped. Wait briefly so a successfully completed command is not
            # recorded as failed with an unresolved exit code.
            record.exit_code = record.proc.poll()
            if record.exit_code is None:
                try:
                    record.exit_code = record.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    record.exit_code = record.proc.poll()
            record.finished_at = time.time()
            record.status = "completed" if record.exit_code == 0 else "failed"
            try:
                os.close(record.master_fd)
            except OSError:
                pass

    def _append(self, record: ProcessRecord, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace")
        with self._lock:
            record.output.append(text)
            record.output_chars += len(text)
            while record.output and record.output_chars > _OUTPUT_LIMIT:
                record.output_chars -= len(record.output.popleft())

    def read(self, process_id: str) -> dict:
        record = self._records[process_id]
        if record.status == "running" and record.proc is not None:
            exit_code = record.proc.poll()
            if exit_code is not None:
                # The PTY reader runs in the background. Wait for it to drain
                # output before reporting a completed process to callers.
                pump_thread = record.pump_thread
                if pump_thread is not None and pump_thread is not threading.current_thread():
                    pump_thread.join(timeout=1)
                if record.status == "running":
                    record.exit_code = exit_code
                    record.finished_at = record.finished_at or time.time()
                    record.status = "completed" if exit_code == 0 else "failed"
        with self._lock:
            output = redact_secrets("".join(record.output))
        data = record.snapshot()
        data["output"] = output
        return data

    def write(self, process_id: str, text: str) -> dict:
        record = self._records[process_id]
        if record.status != "running" or record.master_fd is None:
            raise ProcessPolicyError("Process is not running.")
        os.write(record.master_fd, text.encode("utf-8"))
        return record.snapshot()

    def stop(self, process_id: str) -> dict:
        record = self._records[process_id]
        if record.proc and record.proc.poll() is None:
            os.killpg(record.proc.pid, signal.SIGTERM)
            try:
                record.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(record.proc.pid, signal.SIGKILL)
        record.status = "stopped"
        record.finished_at = time.time()
        record.exit_code = record.proc.poll() if record.proc else None
        return record.snapshot(include_output=True)

    def cleanup_task(self, task_id: str) -> None:
        for record in list(self._records.values()):
            if record.task_id == task_id and record.status == "running":
                self.stop(record.process_id)


_MANAGER = HostProcessManager()


def get_process_manager() -> HostProcessManager:
    return _MANAGER
