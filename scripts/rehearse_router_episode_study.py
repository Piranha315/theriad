#!/usr/bin/env python3
"""Run the zero-network synthetic study regression and retain its artifacts.

Requires the project's pytest development dependencies. Provider replies and
provider prices are synthetic; the coding sandbox can be real. All statistical results
are plumbing checks, never scientific evidence or promotion authorization.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path,
                        help="New empty directory for clearly labeled rehearsal artifacts")
    parser.add_argument("--real-code-sandbox", action="store_true",
                        help="Require the real pinned macOS grader; fail if unavailable")
    parser.add_argument('--profile', choices=('research', 'engineering_canary'), default='research',
                        help='Exercise the selected profile using explicit synthetic fixtures and providers')
    args = parser.parse_args()
    destination = args.directory.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error("--directory must be a new or empty directory; prior studies are immutable")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["THERIAD_EPISODE_REHEARSAL_DIRECTORY"] = str(destination)
    environment["THERIAD_EPISODE_REHEARSAL_REAL_SANDBOX"] = "1" if args.real_code_sandbox else "0"
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/unit/routing/test_episode_rehearsal.py", "-q", "--tb=short",
        "--show-capture=no", '-k', args.profile,
    ], cwd=root, env=environment, check=False)
    if result.returncode == 0:
        print(f"Synthetic offline rehearsal passed. Report: {destination / 'REPORT.md'}")
        print("Real provider spend: $0. No external network or production changes. Code sandbox "
              + ("real." if args.real_code_sandbox else "mocked."))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
