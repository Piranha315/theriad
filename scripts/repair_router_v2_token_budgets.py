#!/usr/bin/env python3
"""Create a new Router V2 corpus with the frozen MC token policy applied.

The source corpus is read-only.  This command refuses to overwrite an output
artifact and changes only the explicit token-limit fields for multiple-choice
rows, including their grader policy metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import (  # noqa: E402
    FORMAT_TOKEN_BUDGETS,
    STUDY_SPLIT_COUNTS,
    corpus_checksum,
    load_jsonl,
)

EXPECTED_SOURCE_CHECKSUM = (
    "0af19ebc3072ee26b365dbe583418651cf28be4f5dfd11e14fd97894d33919b8"
)
TOKEN_POLICY_SOURCE = "router-v2-format-policy-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _without_allowed_token_fields(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if result.get("task_format") == "multiple_choice":
        result.pop("expected_output_tokens", None)
        result.pop("max_output_tokens", None)
        grader = result.get("grader")
        if isinstance(grader, dict):
            grader.pop("token_limit_policy", None)
    return result


def _repair_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("task_format") != "multiple_choice":
        return copy.deepcopy(row)
    budget = FORMAT_TOKEN_BUDGETS["multiple_choice"]
    repaired = copy.deepcopy(row)
    repaired["expected_output_tokens"] = int(budget["expected_output_tokens"])
    repaired["max_output_tokens"] = int(budget["max_output_tokens"])
    grader = repaired.get("grader")
    if not isinstance(grader, dict):
        raise ValueError(f"multiple-choice row {row.get('prompt_id')!r} has no grader object")
    grader["token_limit_policy"] = {
        "expected_output_tokens": int(budget["expected_output_tokens"]),
        "max_output_tokens": int(budget["max_output_tokens"]),
        "source": TOKEN_POLICY_SOURCE,
    }
    repaired["grader"] = grader
    return repaired


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_repaired_corpus(
    source: Path,
    destination: Path,
    *,
    expected_source_checksum: str = EXPECTED_SOURCE_CHECKSUM,
) -> dict[str, Any]:
    source_rows = load_jsonl(source)
    actual_source_checksum = corpus_checksum(source_rows)
    if actual_source_checksum != expected_source_checksum:
        raise ValueError(
            "source corpus checksum does not match the preserved historical corpus: "
            f"expected {expected_source_checksum}, got {actual_source_checksum}"
        )
    raw_rows = _read_rows(source)
    if len(raw_rows) != len(source_rows):
        raise ValueError("source JSONL row count changed while loading")
    if len(raw_rows) != sum(STUDY_SPLIT_COUNTS.values()):
        raise ValueError(f"expected 2,000 source rows, got {len(raw_rows)}")

    repaired_rows = [_repair_row(row) for row in raw_rows]
    multiple_choice_count = sum(
        row.get("task_format") == "multiple_choice" for row in raw_rows
    )
    if multiple_choice_count != 600:
        raise ValueError(f"expected exactly 600 multiple-choice rows, got {multiple_choice_count}")

    for before, after, loaded in zip(raw_rows, repaired_rows, source_rows, strict=True):
        if _without_allowed_token_fields(before) != _without_allowed_token_fields(after):
            raise ValueError(f"repair changed non-token metadata for {loaded.prompt_id}")
        if str(before.get("normalized_prompt_sha256") or "") != loaded.normalized_hash:
            raise ValueError(f"source prompt hash is not frozen for {loaded.prompt_id}")
        if str(after.get("normalized_prompt_sha256") or "") != loaded.normalized_hash:
            raise ValueError(f"repair changed prompt hash for {loaded.prompt_id}")

    encoded = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in repaired_rows
    )
    _write_new(destination, encoded)

    repaired_loaded = load_jsonl(destination)
    repaired_checksum = corpus_checksum(repaired_loaded)
    return {
        "status": "created",
        "source_path": str(source),
        "source_file_sha256": _file_sha256(source),
        "source_corpus_checksum": actual_source_checksum,
        "destination_path": str(destination),
        "destination_file_sha256": _file_sha256(destination),
        "destination_corpus_checksum": repaired_checksum,
        "row_count": len(repaired_loaded),
        "multiple_choice_rows_repaired": multiple_choice_count,
        "changed_fields": [
            "expected_output_tokens",
            "max_output_tokens",
            "grader.token_limit_policy",
        ],
        "token_limit_policy": {
            "expected_output_tokens": int(FORMAT_TOKEN_BUDGETS["multiple_choice"]["expected_output_tokens"]),
            "max_output_tokens": int(FORMAT_TOKEN_BUDGETS["multiple_choice"]["max_output_tokens"]),
            "source": TOKEN_POLICY_SOURCE,
        },
        "prompt_hashes_preserved": True,
        "historical_source_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = build_repaired_corpus(args.source, args.destination)
        if args.report is not None:
            _write_new(
                args.report,
                (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"router-v2 token-policy repair stopped safely: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
