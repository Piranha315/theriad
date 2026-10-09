#!/usr/bin/env python3
"""Freeze exact study/public-anchor overlap hashes and split mapping."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.public_anchors import load_anchor_jsonl  # noqa: E402
from theriad.routing.study import corpus_checksum, load_jsonl  # noqa: E402
from theriad.routing.study_anchor_boundary import (  # noqa: E402
    PUBLIC_ANCHOR_OVERLAP_POLICY,
    STUDY_ANCHOR_BOUNDARY_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_hash"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-overlap-count", type=int, default=500)
    args = parser.parse_args()
    try:
        study = load_jsonl(args.study)
        anchors = load_anchor_jsonl(args.anchors)
        anchor_manifest = json.loads(args.anchor_manifest.read_text(encoding="utf-8"))
        study_by_hash: dict[str, list] = defaultdict(list)
        anchor_by_hash: dict[str, list] = defaultdict(list)
        for row in study:
            study_by_hash[row.normalized_hash].append(row)
        for anchor in anchors:
            anchor_by_hash[anchor.normalized_prompt_hash].append(anchor)
        overlap = sorted(set(study_by_hash) & set(anchor_by_hash))
        if len(overlap) != args.expected_overlap_count:
            raise ValueError(
                f"expected {args.expected_overlap_count} exact overlaps, found {len(overlap)}"
            )
        mapping = []
        for digest in overlap:
            mapping.append(
                {
                    "normalized_prompt_sha256": digest,
                    "study_prompt_ids": sorted(row.prompt_id for row in study_by_hash[digest]),
                    "study_splits": sorted({row.split for row in study_by_hash[digest]}),
                    "anchor_prompt_ids": sorted(row.prompt_id for row in anchor_by_hash[digest]),
                    "anchor_splits": sorted({row.split for row in anchor_by_hash[digest]}),
                }
            )
        payload = {
            "schema_version": STUDY_ANCHOR_BOUNDARY_SCHEMA_VERSION,
            "policy": PUBLIC_ANCHOR_OVERLAP_POLICY,
            "study_path": str(args.study),
            "study_row_count": len(study),
            "study_corpus_checksum": corpus_checksum(study),
            "anchor_path": str(args.anchors),
            "anchor_row_count": len(anchors),
            "anchor_file_sha256": _sha256(args.anchors),
            "anchor_manifest_path": str(args.anchor_manifest),
            "anchor_manifest_sha256": _sha256(args.anchor_manifest),
            "anchor_dataset_id": anchor_manifest.get("dataset_id"),
            "anchor_revision": anchor_manifest.get("revision")
            or anchor_manifest.get("dataset_revision"),
            "overlap_count": len(overlap),
            "overlap_hashes": overlap,
            "mapping": mapping,
            "by_study_split": dict(
                sorted(Counter(row.split for digest in overlap for row in study_by_hash[digest]).items())
            ),
            "by_anchor_split": dict(
                sorted(Counter(anchor.split for digest in overlap for anchor in anchor_by_hash[digest]).items())
            ),
            "excluded_anchor_prompt_ids": sorted(
                anchor.prompt_id for digest in overlap for anchor in anchor_by_hash[digest]
            ),
            "public_anchor_outcomes_used_for_study": False,
            "provider_calls": 0,
            "provider_spend_usd": 0.0,
        }
        payload["manifest_hash"] = _manifest_hash(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"router-v2 anchor-overlap manifest stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
