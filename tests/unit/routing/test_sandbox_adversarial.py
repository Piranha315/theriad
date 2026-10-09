from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from theriad.routing.sandbox import (
    SandboxPolicyError,
    SandboxUnavailable,
    default_policy,
    run_isolated,
    runtime_details,
    runtime_supported,
    sandbox_available,
)

pytestmark = pytest.mark.skipif(
    not sandbox_available(), reason="approved macOS sandbox-exec is unavailable"
)


def _run(code: str, *, policy=None, task: Path | None = None):
    with TemporaryDirectory(prefix="theriad-router-v2-adversarial-", dir="/tmp") as directory:
        task_directory = task or Path(directory)
        return run_isolated(
            (sys.executable, "-c", code),
            task_directory=task_directory,
            policy=policy or default_policy(),
        )


def test_default_deny_blocks_outside_canary_and_home_env_reads(tmp_path: Path):
    outside = tmp_path / "outside-canary.txt"
    outside.write_text("host-secret", encoding="utf-8")
    result = _run(
        "from pathlib import Path\n"
        f"p=Path({str(outside)!r})\n"
        "try:\n"
        "    print(p.read_text())\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    assert result.passed
    assert "host-secret" not in result.stdout
    assert "PermissionError" in result.stdout or "OSError" in result.stdout


def test_network_and_parent_process_creation_are_blocked():
    result = _run(
        "import os, socket, subprocess, sys\n"
        "out=[]\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=0.2)\n"
        "except Exception as exc:\n"
        "    out.append('network:'+type(exc).__name__)\n"
        "try:\n"
        "    os.fork()\n"
        "except Exception as exc:\n"
        "    out.append('fork:'+type(exc).__name__)\n"
        "print(' '.join(out))\n"
    )
    assert result.passed
    assert "network:" in result.stdout
    assert "fork:" in result.stdout


def test_timeout_and_output_limits_are_enforced():
    timeout_policy = replace(default_policy(), wall_clock_seconds=0.5, cpu_seconds=1)
    timeout = _run("while True: pass", policy=timeout_policy)
    assert timeout.timed_out or timeout.status == "timed_out"

    flood_policy = replace(default_policy(), output_bytes=4096)
    flood = _run("print('x' * 100_000)", policy=flood_policy)
    assert flood.status == "output_limit_exceeded"


def test_cpu_limit_is_enforced_before_wall_clock_limit():
    policy = replace(default_policy(), cpu_seconds=1, wall_clock_seconds=3.0)
    result = _run("while True: pass", policy=policy)
    assert result.status == "failed"
    assert result.timed_out is False
    assert result.duration_ms < 2500


def test_task_directory_writes_are_allowed_but_parent_env_is_clean():
    with TemporaryDirectory(prefix="theriad-router-v2-task-", dir="/tmp") as directory:
        task = Path(directory)
        old = os.environ.get("THERIAD_SECRET_CANARY")
        os.environ["THERIAD_SECRET_CANARY"] = "must-not-cross"
        try:
            result = run_isolated(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path\n"
                    "Path('task-output.txt').write_text('ok')\n"
                    "import os\n"
                    "print(os.environ.get('THERIAD_SECRET_CANARY', '<absent>'))\n",
                ),
                task_directory=task,
                policy=default_policy(),
            )
        finally:
            if old is None:
                os.environ.pop("THERIAD_SECRET_CANARY", None)
            else:
                os.environ["THERIAD_SECRET_CANARY"] = old
        assert result.passed
        assert (task / "task-output.txt").read_text(encoding="utf-8") == "ok"
        assert "must-not-cross" not in result.stdout
        assert "<absent>" in result.stdout


def test_symlinked_venv_launcher_and_canonical_python_app_helper_are_admitted():
    launcher = Path(sys.executable)
    if launcher.resolve() == launcher:
        pytest.skip("active runtime is not a symlinked virtual-environment launcher")
    details = runtime_details()
    helper = details.get("python_app_helper")
    if not isinstance(helper, str):
        pytest.skip("active runtime has no Python.app helper")
    policy = default_policy()
    assert runtime_supported()
    assert str(launcher) in policy.runtime_allowlist
    assert str(launcher.resolve()) in policy.runtime_allowlist
    assert helper in policy.runtime_allowlist
    with TemporaryDirectory(prefix="theriad-router-v2-helper-", dir="/tmp") as directory:
        result = run_isolated(
            (helper, "-c", "print('python-app-helper-ok')"),
            task_directory=Path(directory),
            policy=policy,
        )
    assert result.passed
    assert "python-app-helper-ok" in result.stdout


def test_missing_or_incompatible_pinned_runtime_fails_closed(tmp_path: Path):
    missing = tmp_path / "missing-python"
    assert runtime_supported(missing) is False
    with pytest.raises(SandboxUnavailable):
        run_isolated(
            (str(missing), "-c", "print('must-not-run')"),
            task_directory=tmp_path,
            policy=default_policy(runtime=str(missing)),
        )

    details = runtime_details()
    if not details.get("python_app_helper"):
        pytest.skip("active runtime has no Python.app helper to test policy omission")
    incompatible = replace(default_policy(), runtime_allowlist=(str(Path(sys.executable)),))
    with pytest.raises(SandboxPolicyError):
        run_isolated(
            (sys.executable, "-c", "print('must-not-run')"),
            task_directory=tmp_path,
            policy=incompatible,
        )
