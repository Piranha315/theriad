"""Fail-closed execution boundary for Router V2 code graders.

This module is deliberately small and provider-free.  It never falls back to
``subprocess`` on the host: a run is accepted only when the selected sandbox
backend is present and the command, runtime, environment, and limits satisfy
the frozen policy.  The current macOS backend uses ``sandbox-exec`` with a
network-deny/read-only-base policy and a private writable task directory.

The acquisition workflow may run the harmless policy fixture.  Benchmark or
candidate code is only executed through :func:`run_python_tests`, which keeps
the code and trusted test harness in the temporary task directory and invokes
the sandbox backend.
"""
from __future__ import annotations

import json
import os
import platform
import re
import selectors
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SANDBOX_POLICY_VERSION = "router-v2-code-sandbox-v2-default-deny"
SANDBOX_EXECUTOR_ID = "router-v2.macos.sandbox-exec"
CODE_GRADER_RUNTIME_ENV = "THERIAD_ROUTER_V2_CODE_GRADER_PYTHON"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_CPU_SECONDS = 8
DEFAULT_MEMORY_BYTES = 768 * 1024 * 1024
DEFAULT_PROCESS_LIMIT = 32
DEFAULT_OUTPUT_BYTES = 256 * 1024
# Code collection and validation use an explicitly pinned runtime.  The
# sandbox still admits its symlink, canonical interpreter, and Python.app
# helper as appropriate, but never silently selects a different environment.
SUPPORTED_RUNTIME_APPROACH = "B_pinned_code_grader_runtime_fail_closed"


class SandboxUnavailable(RuntimeError):
    """Raised when no approved isolated execution backend is available."""


class SandboxPolicyError(ValueError):
    """Raised when a command cannot be admitted by the frozen policy."""


@dataclass(frozen=True)
class SandboxPolicy:
    """The security properties required for one isolated task execution."""

    version: str = SANDBOX_POLICY_VERSION
    executor_id: str = SANDBOX_EXECUTOR_ID
    network_disabled: bool = True
    base_filesystem_read_only: bool = True
    task_directory_only_writable: bool = True
    credentials_in_environment: bool = False
    posix_shared_memory_for_runtime: bool = True
    language_allowlist: tuple[str, ...] = ("python",)
    runtime_allowlist: tuple[str, ...] = ()
    allowed_read_paths: tuple[str, ...] = ()
    python_home: str | None = None
    python_path: tuple[str, ...] = ()
    cpu_seconds: int = DEFAULT_CPU_SECONDS
    memory_bytes: int = DEFAULT_MEMORY_BYTES
    process_limit: int = DEFAULT_PROCESS_LIMIT
    wall_clock_seconds: float = DEFAULT_TIMEOUT_SECONDS
    output_bytes: int = DEFAULT_OUTPUT_BYTES


@dataclass(frozen=True)
class SandboxResult:
    """A bounded execution result with no implicit success conversion."""

    status: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    executor_id: str
    policy_version: str

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "executor_id": self.executor_id,
            "policy_version": self.policy_version,
        }


def _canonical_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _launcher_path(runtime: str | Path | None) -> Path:
    selected = Path(runtime or sys.executable).expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return selected


def code_grader_runtime(*, required: bool = False) -> str | None:
    """Return the configured code-grader interpreter or fail closed.

    Low-level sandbox fixtures may inspect the active interpreter by leaving
    ``required`` false.  Actual executable grading uses ``required=True`` so
    an unpinned host runtime can never receive benchmark code.
    """

    configured = os.environ.get(CODE_GRADER_RUNTIME_ENV, "").strip()
    if configured:
        return configured
    if required:
        return None
    return sys.executable


def _python_app_helper(runtime: Path) -> Path | None:
    """Return the exact Python.app helper used by framework Python builds."""

    canonical = Path(_canonical_path(runtime))
    if canonical.parent.name != "bin":
        return None
    helper = canonical.parent.parent / "Resources/Python.app/Contents/MacOS/Python"
    return helper if helper.is_file() else None


def _sandbox_exec_path() -> str | None:
    candidate = "/usr/bin/sandbox-exec"
    return candidate if Path(candidate).is_file() and os.access(candidate, os.X_OK) else None


def _runtime_process_identity(runtime: Path) -> dict[str, str]:
    """Read identity from the selected interpreter, never from the host."""

    try:
        completed = subprocess.run(
            (
                str(runtime),
                "-c",
                "import json, os, platform, sys; "
                "print(json.dumps({'sys_executable': sys.executable, "
                "'resolved_executable': os.path.realpath(sys.executable), "
                "'sys_version': sys.version, 'platform': platform.platform()}, sort_keys=True))",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": str(runtime.parent.parent),
                "PYTHONDONTWRITEBYTECODE": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
        value = json.loads(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError) as exc:
        raise SandboxUnavailable(f"could not inspect selected runtime {runtime}: {exc}") from exc
    if not isinstance(value, dict):
        raise SandboxUnavailable(f"selected runtime identity is malformed: {runtime}")
    return {str(key): str(item) for key, item in value.items()}


def sandbox_available() -> bool:
    """Return whether the approved local backend is available."""

    return _sandbox_exec_path() is not None and sys.platform == "darwin"


def _virtualenv_python_home(runtime: Path) -> Path | None:
    """Resolve the base Python home recorded by a virtualenv launcher."""

    config = runtime.parent.parent / "pyvenv.cfg"
    if not config.is_file():
        return None
    try:
        recorded_executable: Path | None = None
        recorded_home: Path | None = None
        for line in config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if not separator:
                continue
            if key.strip() == "executable":
                recorded_executable = Path(value.strip()).expanduser()
            elif key.strip() == "home":
                recorded_home = Path(value.strip()).expanduser()
        if recorded_executable is not None:
            executable = recorded_executable.resolve()
            if executable.parent.name == "bin":
                return executable.parent.parent
        if recorded_home is not None:
            if recorded_home.name == "bin":
                recorded_home = recorded_home.parent
            return recorded_home.resolve()
    except (OSError, RuntimeError):
        return None
    return None


def _virtualenv_site_packages(runtime: Path) -> tuple[Path, ...]:
    """Find only the selected virtualenv's versioned package directory."""

    root = runtime.parent.parent
    config = root / "pyvenv.cfg"
    if not config.is_file():
        return ()
    version = ""
    try:
        for line in config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "version_info":
                version = value.strip().split(".", 2)[0:2]
                break
    except OSError:
        return ()
    if isinstance(version, list) and len(version) == 2:
        candidate = root / "lib" / f"python{version[0]}.{version[1]}" / "site-packages"
        return (candidate,) if candidate.is_dir() else ()
    candidates = tuple(sorted((root / "lib").glob("python*/site-packages"))) if (root / "lib").is_dir() else ()
    return tuple(path for path in candidates if path.is_dir())


def _default_read_paths(
    runtime: Path,
    extra: Sequence[str | Path] = (),
    *,
    python_home: Path | None = None,
    runtime_paths: Sequence[str | Path] = (),
) -> tuple[str, ...]:
    """Return explicitly approved runtime/library paths for default-deny read."""

    paths: list[str] = [
        str(runtime),
        "/bin",
        "/usr/bin",
        "/sbin",
        "/usr/sbin",
        "/usr/lib",
        "/usr/libexec",
        "/System/Library",
        "/Library/Frameworks",
        "/dev",
    ]
    paths.extend(str(Path(value).expanduser().resolve()) for value in runtime_paths)
    if python_home is not None:
        paths.append(str(python_home))
        paths.append(str(python_home / "lib"))
    try:
        paths.extend(
            str(Path(value).resolve())
            for value in sysconfig.get_paths().values()
            if value
        )
    except (OSError, RuntimeError):
        pass
    # A virtualenv runtime needs its sibling library tree.  This is an
    # explicit approved dependency root, not a permission to read the user's
    # home directory.
    if runtime.parent.name in {"bin", "Scripts"}:
        paths.append(str(runtime.parent.parent))
    paths.extend(str(Path(value).expanduser().resolve()) for value in extra)
    unique: list[str] = []
    for value in paths:
        if value not in unique and Path(value).exists():
            unique.append(value)
    return tuple(unique)


def default_policy(
    *, runtime: str | None = None, allowed_read_paths: Sequence[str | Path] = ()
) -> SandboxPolicy:
    """Return the policy with an explicit runtime, if one is supplied."""

    selected_path = _launcher_path(runtime)
    selected = _canonical_path(selected_path)
    canonical_path = Path(selected)
    python_home = _virtualenv_python_home(selected_path)
    site_packages = _virtualenv_site_packages(selected_path)
    runtime_allowlist = [str(selected_path), selected]
    helper = _python_app_helper(canonical_path)
    if helper is not None:
        runtime_allowlist.append(str(helper))
    return SandboxPolicy(
        runtime_allowlist=tuple(dict.fromkeys(runtime_allowlist)),
        allowed_read_paths=_default_read_paths(
            selected_path,
            allowed_read_paths,
            python_home=python_home,
            runtime_paths=(*runtime_allowlist, *site_packages),
        ),
        python_home=str(python_home) if python_home is not None else None,
        python_path=tuple(str(path) for path in site_packages),
    )


def runtime_details(runtime: str | Path | None = None) -> dict[str, object]:
    """Describe and validate the selected code-grader runtime."""

    launcher = _launcher_path(runtime)
    resolved = Path(_canonical_path(launcher))
    helper = _python_app_helper(resolved)
    process_exec_allowlist = [str(launcher), str(resolved)]
    if helper is not None:
        process_exec_allowlist.append(str(helper))
    python_home = _virtualenv_python_home(launcher)
    reasons: list[str] = []
    try:
        identity = _runtime_process_identity(launcher)
    except SandboxUnavailable as exc:
        identity = {}
        reasons.append(str(exc))
    if sys.platform != "darwin":
        reasons.append("macOS sandbox-exec backend requires Darwin")
    if _sandbox_exec_path() is None:
        reasons.append("/usr/bin/sandbox-exec is unavailable")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        reasons.append("resolved interpreter is missing or not executable")
    if helper is not None and not os.access(helper, os.X_OK):
        reasons.append("Python.app helper is missing or not executable")
    if python_home is not None and not python_home.is_dir():
        reasons.append("virtual-environment Python home is missing")
    return {
        "approach": SUPPORTED_RUNTIME_APPROACH,
        "selection": "active_process_interpreter" if runtime is None else "explicit_runtime",
        "configured_runtime_env": CODE_GRADER_RUNTIME_ENV,
        "requested_executable": str(launcher),
        "sys_executable": identity.get("sys_executable", sys.executable),
        "sys_version": identity.get("sys_version", sys.version),
        "resolved_executable": str(resolved),
        "python_app_helper": str(helper) if helper is not None else None,
        "python_home": str(python_home) if python_home is not None else None,
        "python_path": [str(path) for path in _virtualenv_site_packages(launcher)],
        "process_exec_allowlist": list(dict.fromkeys(process_exec_allowlist)),
        "platform": platform.platform(),
        "platform_system": sys.platform,
        "sandbox_exec": _sandbox_exec_path(),
        "supported": not reasons,
        "reasons": reasons,
    }


def runtime_supported(runtime: str | Path | None = None) -> bool:
    """Return whether the selected runtime can be admitted fail-closed."""

    return bool(runtime_details(runtime)["supported"])


def _profile_for_task(task_directory: Path, policy: SandboxPolicy) -> str:
    """Build the macOS profile for one exact private writable directory."""

    task = _canonical_path(task_directory)
    read_paths = list(policy.allowed_read_paths) + [task]
    literals = [f"(subpath {json.dumps(path)})" for path in read_paths]
    runtimes = [f"(literal {json.dumps(path)})" for path in policy.runtime_allowlist]
    parent_literals: set[str] = {"/"}
    for value in read_paths:
        current = Path(value)
        while current != current.parent:
            parent_literals.add(str(current.parent))
            current = current.parent
    parent_rules = [
        f"(allow file-read* (literal {json.dumps(path)}))"
        for path in sorted(parent_literals, key=lambda item: (item.count("/"), item))
        if path != "/"
    ]
    # This is intentionally deny-default.  The old `(allow default)` made the
    # read-only claim false: candidate code could open any host file.  Process
    # execution is restricted to the selected interpreter and process-fork is
    # left denied so a candidate cannot create an unbounded child tree.
    return "\n".join(
        (
            "(version 1)",
            "(deny default)",
            # The literal root permission is only for path traversal.  File
            # contents remain limited to the explicit subpaths below.
            '(allow file-read* (literal "/"))',
            *parent_rules,
            *[f"(allow file-read* {literal})" for literal in literals],
            f"(allow file-write* (subpath {json.dumps(task)}))",
            f"(allow process-exec {' '.join(runtimes)})",
            "(allow sysctl-read)",
            *(
                (
                    # NumPy/BLAS wheels bundled with the pinned benchmark
                    # runtimes use POSIX shared memory for their single-process
                    # initialization. It is kernel IPC, not host filesystem or
                    # network access; process fork remains denied and the task
                    # is still run sequentially.
                    "(allow ipc-posix-shm*)",
                )
                if policy.posix_shared_memory_for_runtime
                else ()
            ),
            "(deny network*)",
        )
    )


def _bounded_environment(task_directory: Path, policy: SandboxPolicy) -> dict[str, str]:
    """Return an allowlisted environment with no inherited credentials."""

    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": _canonical_path(task_directory),
        "TMPDIR": _canonical_path(task_directory),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Keep numerical benchmark dependencies inside the private task
        # directory and avoid OpenMP shared-memory initialization outside the
        # allowlisted filesystem.  These limits also make grader timing
        # reproducible across runs.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "KMP_USE_SHM": "0",
        "KMP_SHM_DIR": _canonical_path(task_directory),
    }
    if policy.python_home is not None:
        environment["PYTHONHOME"] = policy.python_home
    if policy.python_path:
        environment["PYTHONPATH"] = os.pathsep.join(policy.python_path)
    return environment


def _set_limits(policy: SandboxPolicy) -> None:
    """Apply child resource limits where macOS exposes them."""

    import resource

    for limit, value in (
        (resource.RLIMIT_CPU, policy.cpu_seconds),
        (resource.RLIMIT_AS, policy.memory_bytes),
        (resource.RLIMIT_FSIZE, policy.output_bytes),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            # macOS exposes some limits but rejects others for a child.  The
            # sandbox profile and parent watchdog remain mandatory controls.
            pass
    if hasattr(resource, "RLIMIT_NPROC"):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (policy.process_limit, policy.process_limit))
        except (OSError, ValueError):
            pass


def _validate_command(
    argv: Sequence[str], *, language: str, policy: SandboxPolicy
) -> tuple[str, ...]:
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise SandboxPolicyError("sandbox command must be a non-empty argv sequence")
    if language not in policy.language_allowlist:
        raise SandboxPolicyError(f"language {language!r} is not allowlisted")
    runtime = _canonical_path(Path(argv[0]))
    allowed = tuple(_canonical_path(Path(value)) for value in policy.runtime_allowlist)
    if runtime not in allowed:
        raise SandboxPolicyError(f"runtime {runtime!r} is not allowlisted")
    # macOS sandbox-exec does not reliably admit execution through a symlinked
    # virtualenv launcher.  Execute the canonical target after the admission
    # check; default_policy carries the virtualenv's recorded PYTHONHOME so
    # the interpreter still loads the pinned runtime and site paths.
    return (runtime, *argv[1:])


def _decode_limited(value: bytes, limit: int) -> str:
    if len(value) <= limit:
        return value.decode("utf-8", errors="replace")
    return value[:limit].decode("utf-8", errors="replace") + "\n[output truncated]"


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _bounded_communicate(
    process: subprocess.Popen[bytes], *, stdin: str, timeout: float, output_limit: int
) -> tuple[bytes, bytes, bool, bool]:
    """Drain both pipes without allowing an output flood to grow unbounded."""

    if process.stdin is not None:
        process.stdin.write(stdin.encode("utf-8"))
        process.stdin.close()
    selector = selectors.DefaultSelector()
    streams = {process.stdout: bytearray(), process.stderr: bytearray()}
    for stream in streams:
        if stream is not None:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    timed_out = False
    output_limited = False
    killed = False
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            timed_out = True
            _kill_process_group(process)
            killed = True
        for key, _ in selector.select(max(0.0, min(remaining, 0.05))):
            stream = key.fileobj
            try:
                chunk = os.read(stream.fileno(), 8192)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(stream)
                continue
            buffer = streams[stream]
            buffer.extend(chunk)
            if len(buffer) > output_limit and not output_limited:
                output_limited = True
                if process.poll() is None:
                    _kill_process_group(process)
                    killed = True
        if process.poll() is not None and not killed:
            # Keep draining until EOF so the child cannot leave a pipe-backed
            # descendant holding the collector open.
            continue
        if killed and process.poll() is not None:
            continue
    process.wait()
    return (
        bytes(streams.get(process.stdout, b"")),
        bytes(streams.get(process.stderr, b"")),
        timed_out,
        output_limited,
    )


def run_isolated(
    argv: Sequence[str],
    *,
    task_directory: Path,
    language: str = "python",
    stdin: str = "",
    policy: SandboxPolicy | None = None,
) -> SandboxResult:
    """Run an allowlisted command only inside the approved sandbox backend."""

    selected_policy = policy or default_policy(runtime=argv[0] if argv else None)
    task_directory = Path(task_directory)
    task_directory.mkdir(parents=True, exist_ok=True)
    if not sandbox_available():
        raise SandboxUnavailable(
            "macOS sandbox-exec is unavailable; refusing host execution"
        )
    runtime = selected_policy.runtime_allowlist[0] if selected_policy.runtime_allowlist else None
    details = runtime_details(runtime)
    if not details["supported"]:
        raise SandboxUnavailable(
            "selected code-grader runtime is unavailable or incompatible: "
            + "; ".join(str(reason) for reason in details["reasons"])
        )
    helper = details.get("python_app_helper")
    canonical_runtimes = {_canonical_path(Path(value)) for value in selected_policy.runtime_allowlist}
    if helper is not None and _canonical_path(Path(str(helper))) not in canonical_runtimes:
        raise SandboxPolicyError(
            "sandbox policy omits the selected runtime's canonical Python.app helper"
        )
    command = _validate_command(argv, language=language, policy=selected_policy)
    sandbox_exec = _sandbox_exec_path()
    if sandbox_exec is None:  # defensive check after availability probe
        raise SandboxUnavailable("approved sandbox backend disappeared")
    if not (
        selected_policy.network_disabled
        and selected_policy.base_filesystem_read_only
        and selected_policy.task_directory_only_writable
        and not selected_policy.credentials_in_environment
    ):
        raise SandboxPolicyError("sandbox policy is weaker than the frozen contract")
    start = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    try:
        process = subprocess.Popen(
            (sandbox_exec, "-p", _profile_for_task(task_directory, selected_policy), *command),
            cwd=_canonical_path(task_directory),
            env=_bounded_environment(task_directory, selected_policy),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _set_limits(selected_policy),
        )
        stdout, stderr, timed_out, output_limited = _bounded_communicate(
            process,
            stdin=stdin,
            timeout=selected_policy.wall_clock_seconds,
            output_limit=selected_policy.output_bytes,
        )
    finally:
        if process is not None and process.poll() is None:
            _kill_process_group(process)
            process.wait()
    returncode = process.returncode if process is not None else None
    if timed_out:
        status = "timed_out"
    elif output_limited:
        status = "output_limit_exceeded"
    elif returncode == 0:
        status = "passed"
    else:
        status = "failed"
    return SandboxResult(
        status=status,
        returncode=returncode,
        stdout=_decode_limited(stdout, selected_policy.output_bytes),
        stderr=_decode_limited(stderr, selected_policy.output_bytes),
        timed_out=timed_out,
        duration_ms=(time.monotonic() - start) * 1000.0,
        executor_id=selected_policy.executor_id,
        policy_version=selected_policy.version,
    )


def run_python_tests(
    candidate_source: str,
    trusted_test_source: str,
    *,
    entry_point: str,
    policy: SandboxPolicy | None = None,
) -> SandboxResult:
    """Run one candidate/test pair in a fresh sandbox task directory.

    ``trusted_test_source`` is pinned benchmark code and ``candidate_source``
    is untrusted model output.  Neither is executed by the acquisition host.
    The returned result stays pending/failed at the caller's discretion; this
    function never turns a missing or unavailable executor into a grade.
    """

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point):
        raise SandboxPolicyError("entry point must be a simple Python identifier")
    selected_policy = policy
    if selected_policy is None:
        selected_runtime = code_grader_runtime(required=True)
        if selected_runtime is None:
            raise SandboxUnavailable(
                f"{CODE_GRADER_RUNTIME_ENV} is not configured; refusing unpinned code grading"
            )
        selected_policy = default_policy(runtime=selected_runtime)
    runtime = selected_policy.runtime_allowlist[0]
    with tempfile.TemporaryDirectory(prefix="theriad-router-v2-code-") as directory:
        task_directory = Path(directory)
        (task_directory / "solution.py").write_text(candidate_source, encoding="utf-8")
        runner = (
            "import unittest\n"
            "from solution import *  # noqa: F401,F403\n"
            f"{trusted_test_source}\n"
            "if callable(globals().get('check')):\n"
            f"    check({entry_point})\n"
            "elif __name__ == '__main__':\n"
            "    result = unittest.main(verbosity=0, exit=False)\n"
            "    raise SystemExit(0 if result.result.wasSuccessful() else 1)\n"
        )
        (task_directory / "run_tests.py").write_text(runner, encoding="utf-8")
        return run_isolated(
            (runtime, "run_tests.py"),
            task_directory=task_directory,
            language="python",
            policy=selected_policy,
        )


def run_harmless_fixture() -> SandboxResult:
    """Smoke-test network denial and private-directory write access only."""

    if not sandbox_available():
        raise SandboxUnavailable("cannot run sandbox fixture without sandbox-exec")
    policy = default_policy()
    with tempfile.TemporaryDirectory(prefix="theriad-router-v2-sandbox-") as directory:
        task_directory = Path(directory)
        code = (
            "from pathlib import Path\n"
            "Path('fixture.txt').write_text('ok', encoding='utf-8')\n"
            "print(Path('fixture.txt').read_text(encoding='utf-8'))\n"
        )
        return run_isolated(
            (policy.runtime_allowlist[0], "-c", code),
            task_directory=task_directory,
            policy=policy,
        )


__all__ = [
    "CODE_GRADER_RUNTIME_ENV",
    "SANDBOX_EXECUTOR_ID",
    "SANDBOX_POLICY_VERSION",
    "SUPPORTED_RUNTIME_APPROACH",
    "SandboxPolicy",
    "SandboxPolicyError",
    "SandboxResult",
    "SandboxUnavailable",
    "code_grader_runtime",
    "default_policy",
    "run_harmless_fixture",
    "run_isolated",
    "run_python_tests",
    "runtime_details",
    "runtime_supported",
    "sandbox_available",
]
