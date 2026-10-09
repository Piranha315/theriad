"""Bounded browser-agent test runner. It reports failures; it never self-edits."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = min(max(int(os.environ.get("THERIAD_SELFTEST_ATTEMPTS", "1")), 1), 5)
# Default to the interpreter already running this script. Resolving "pytest"
# or "python" from PATH picks up whichever install happens to come first,
# which in a virtualenv checkout is routinely a system interpreter without
# Theriad's dependencies -- the self-test then fails on a bare ImportError
# instead of reporting anything about the agent. The env overrides remain for
# deliberately targeting a different environment.
PYTHON = os.environ.get("THERIAD_SELFTEST_PYTHON") or sys.executable
PYTEST = os.environ.get("THERIAD_SELFTEST_PYTEST")
PYTEST_CMD = [PYTEST] if PYTEST else [PYTHON, "-m", "pytest"]
COMMANDS = [
    [*PYTEST_CMD,
     "tests/unit/api/test_initial_navigation.py",
     "tests/unit/api/test_captcha_search.py",
     "tests/unit/api/test_local_browser_runtime.py", "-q"],
    [PYTHON, "scripts/agent_e2e_smoke.py"],
]


def main() -> int:
    for attempt in range(1, ATTEMPTS + 1):
        print(f"Agent self-test attempt {attempt}/{ATTEMPTS}")
        failed = False
        for command in COMMANDS:
            print("+", " ".join(command))
            result = subprocess.run(command, cwd=ROOT, text=True)
            if result.returncode:
                failed = True
                print(
                    f"FAILED ({result.returncode}): {' '.join(command)}",
                    file=sys.stderr,
                )
                break
        if not failed:
            print("Agent self-test passed.")
            return 0
    print("Agent self-test exhausted its bounded attempts.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
