#!/usr/bin/env python3
"""Bootstrap or verify the pinned Router V2 executor environments.

The default operation is provider-free and read-only with respect to package
installation: it inspects already acquired runtimes, writes frozen dependency
evidence, and fails closed when any required environment is absent.  Rebuild
commands are recorded for an operator to run deliberately; this command never
falls back to an undocumented temporary virtualenv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.executor_manifest import (  # noqa: E402
    DEFAULT_SOURCE_CACHE,
    ExecutorManifestError,
    create_executor_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--code-python", type=Path)
    parser.add_argument("--bfcl-python", type=Path)
    parser.add_argument("--tau-python", type=Path)
    parser.add_argument("--bfcl-source", type=Path)
    parser.add_argument("--bfcl-cache", type=Path)
    parser.add_argument("--tau-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_cache
    values = {
        "code_python": args.code_python or root / "acquired/code-grader-venv/bin/python",
        "bfcl_python": args.bfcl_python or root / "acquired/bfcl-venv/bin/python",
        "tau_python": args.tau_python or root / "acquired/tau2-venv/bin/python",
        "bfcl_source": args.bfcl_source or root / "acquired/bfcl/berkeley-function-call-leaderboard",
        "bfcl_cache": args.bfcl_cache or root / "bfcl-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "tau_source": args.tau_source or root / "acquired/tau2-bench",
    }
    try:
        manifest = create_executor_manifest(
            output=args.output,
            source_cache=root,
            **values,
        )
    except (ExecutorManifestError, OSError, ValueError) as exc:
        print(f"router-v2 executor bootstrap stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    lock_dir = args.output.parent / "executor-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for role, record in (manifest.get("runtimes") or {}).items():
        frozen = record.get("dependency_frozen_set")
        if isinstance(frozen, list):
            lock_dir.joinpath(f"{role}.freeze.txt").write_text(
                "\n".join(str(item) for item in frozen) + "\n", encoding="utf-8"
            )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
