#!/usr/bin/env python3
"""Validate and freeze a Router V2 study corpus supplied as local JSONL.

This command does not fetch benchmark data.  The caller must provide explicit
licence, revision, provenance, split, and grading metadata for every row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import (  # noqa: E402
    GROK_SELECTION_ALGORITHM_VERSION,
    OOD_POLICY_VERSION,
    StudyPrompt,
    audit_corpus,
    corpus_checksum,
    write_jsonl,
)


def _load_lenient(path: Path) -> tuple[list[StudyPrompt], list[dict]]:
    accepted: list[StudyPrompt] = []
    rejected: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                accepted.append(StudyPrompt.from_dict(raw, require_frozen_hash=True))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                rejected.append({"line": line_number, "reason": f"{type(exc).__name__}: {exc}"})
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="candidate prompt JSONL")
    parser.add_argument("--output", type=Path, required=True, help="frozen study JSONL")
    parser.add_argument("--audit", type=Path, required=True, help="audit JSON output")
    parser.add_argument("--rejections", type=Path, required=True, help="rejection JSONL output")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="write a research freeze manifest only after the corpus passes",
    )
    args = parser.parse_args()

    rows, rejected = _load_lenient(args.input)
    audit = audit_corpus(rows, rejected=rejected)
    checksum = corpus_checksum(rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        **audit.to_dict(),
        "corpus_checksum": checksum,
        "candidate_path": str(args.input),
        "ood_policy_version": OOD_POLICY_VERSION,
        "grok_selection_algorithm_version": GROK_SELECTION_ALGORITHM_VERSION,
    }
    args.audit.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.rejections.parent.mkdir(parents=True, exist_ok=True)
    with args.rejections.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not audit.valid:
        print(json.dumps(audit_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    write_jsonl(rows, args.output)
    if args.freeze_manifest:
        manifest = {
            "schema_version": 1,
            "status": "frozen_research_corpus_input_only",
            "manifest_type": "corpus_input_only_not_final_pretest_freeze",
            "test_unlock": False,
            "pretest_manifest_required": True,
            "corpus_path": str(args.output),
            "corpus_checksum": corpus_checksum(rows),
            "row_count": len(rows),
            "split_counts": audit.split_counts,
            "lane_counts": audit.lane_counts,
            "task_format_counts": audit.task_format_counts,
            "source_counts": audit.source_counts,
            "benchmark_counts": audit.benchmark_counts,
            "stratum_counts": audit.stratum_counts,
            "ood_policy_version": OOD_POLICY_VERSION,
            "grok_selection_algorithm_version": GROK_SELECTION_ALGORITHM_VERSION,
            "public_anchor_outcomes_used_as_candidate_outcomes": False,
        }
        args.freeze_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output": str(args.output), **audit.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
