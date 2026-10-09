#!/usr/bin/env python3
"""Verify the pinned public-anchor manifest and local import without network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.public_anchors import load_anchor_jsonl  # noqa: E402

DATASET_ID = "i-DeepSearch/observation-masking-eval-logs"
REVISION = "19342195a5c5c9f8ae016cac6a388d937371df8e"
EXPECTED = {
    "file_count": 113,
    "prompt_count": 1322,
    "observation_count": 77369,
    "explicit_outcome_count": 77368,
    "model_count": 9,
    "protocol_count": 27,
    "rejected_record_count": 0,
}


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    _check(manifest.get("dataset_id") == DATASET_ID, "manifest dataset_id mismatch", failures)
    _check(manifest.get("revision") == REVISION, "manifest revision mismatch", failures)
    _check(audit.get("dataset_id") == DATASET_ID, "audit dataset_id mismatch", failures)
    _check(audit.get("dataset_revision") == REVISION, "audit revision mismatch", failures)

    entries = manifest.get("files")
    _check(isinstance(entries, list), "manifest files is not a list", failures)
    entries = entries if isinstance(entries, list) else []
    _check(len(entries) == EXPECTED["file_count"], f"manifest has {len(entries)} files", failures)
    _check(manifest.get("file_count") == EXPECTED["file_count"], "manifest file_count mismatch", failures)

    verified_files: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append("manifest contains a non-object file entry")
            continue
        relative = str(entry.get("cache_relative_path") or Path("files") / REVISION / str(entry.get("path") or ""))
        target = args.cache_dir / relative
        if not target.is_file():
            failures.append(f"missing cached file: {relative}")
            continue
        actual_hash, actual_size = _sha256(target)
        _check(actual_hash == entry.get("sha256"), f"SHA-256 mismatch: {entry.get('path')}", failures)
        _check(entry.get("size_bytes") is None or actual_size == int(entry["size_bytes"]), f"size mismatch: {entry.get('path')}", failures)
        verified_files.append({"path": entry.get("path"), "sha256": actual_hash})

    manifest_hash, _ = _sha256(args.manifest)
    _check(audit.get("manifest_sha256") == manifest_hash, "audit manifest_sha256 mismatch", failures)
    audit_file_hashes = {
        str(row.get("path")): str(row.get("sha256"))
        for row in audit.get("file_hashes", [])
        if isinstance(row, dict)
    }
    _check(len(audit_file_hashes) == EXPECTED["file_count"], "audit does not contain all file hashes", failures)
    _check(
        {str(row["path"]): str(row["sha256"]) for row in verified_files} == audit_file_hashes,
        "audit file hashes differ from verified cache hashes",
        failures,
    )

    anchors = load_anchor_jsonl(args.anchors)
    observations = [observation for anchor in anchors for observation in anchor.observations]
    actual = {
        "file_count": len(entries),
        "prompt_count": len(anchors),
        "observation_count": len(observations),
        "explicit_outcome_count": sum(item.outcome is not None for item in observations),
        "model_count": len({item.model_id for item in observations}),
        "protocol_count": len({item.protocol_id for item in observations}),
        "rejected_record_count": int(audit.get("rejected_record_count", -1)),
    }
    for key, expected in EXPECTED.items():
        _check(actual[key] == expected, f"{key}: expected {expected}, got {actual[key]}", failures)
    _check(args.rejections.read_text(encoding="utf-8") == "", "rejection file is not empty", failures)

    report = {
        "status": "passed" if not failures else "failed",
        "provider_calls": 0,
        "dataset_id": DATASET_ID,
        "dataset_revision": REVISION,
        "file_hashes_verified": len(verified_files),
        "counts": actual,
        "manifest_sha256": manifest_hash,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
