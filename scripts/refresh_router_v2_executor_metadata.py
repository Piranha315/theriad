#!/usr/bin/env python3
"""Rebind frozen executable-code rows to the current local executor contract.

This is a metadata-only corpus migration.  It refuses to change prompt text,
normalized prompt hashes, split assignments, source provenance, or row count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.sandbox import SANDBOX_POLICY_VERSION  # noqa: E402
from theriad.routing.study import (  # noqa: E402
    CODE_LANE_FORMATS,
    StudyPrompt,
    corpus_checksum,
    load_jsonl,
)
from theriad.routing.study_adapters import CODE_ADAPTER_VERSION  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate(input_path: Path, output_path: Path) -> dict[str, object]:
    raw_rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    before_rows = tuple(StudyPrompt.from_dict(row, require_frozen_hash=True) for row in raw_rows)
    before_hashes = tuple(row.normalized_hash for row in before_rows)
    before_corpus = corpus_checksum(before_rows)
    changed = 0
    output_rows: list[dict] = []
    for raw, row in zip(raw_rows, before_rows, strict=True):
        updated = dict(raw)
        if row.task_format in CODE_LANE_FORMATS:
            grader = dict(updated.get("grader") or {})
            grader["adapter_version"] = CODE_ADAPTER_VERSION
            grader["sandbox_policy_version"] = SANDBOX_POLICY_VERSION
            updated["grader"] = grader
            changed += 1
        output_rows.append(updated)
    if changed != 300:
        raise ValueError(f"expected exactly 300 executable-code rows, found {changed}")
    if output_path.resolve() == input_path.resolve():
        _write_atomic(output_path, output_rows)
    else:
        _write_atomic(output_path, output_rows)
    after_rows = load_jsonl(output_path)
    after_hashes = tuple(row.normalized_hash for row in after_rows)
    if len(after_rows) != len(before_rows):
        raise ValueError("metadata migration changed corpus row count")
    if after_hashes != before_hashes:
        raise ValueError("metadata migration changed normalized prompt hashes")
    return {
        "status": "passed",
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(after_rows),
        "executable_code_rows_rebound": changed,
        "prompt_hashes_preserved": True,
        "before_corpus_checksum": before_corpus,
        "after_corpus_checksum": corpus_checksum(after_rows),
        "before_file_sha256": _file_sha256(input_path)
        if input_path.resolve() != output_path.resolve()
        else None,
        "after_file_sha256": _file_sha256(output_path),
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "code_adapter_version": CODE_ADAPTER_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = migrate(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
