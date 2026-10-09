#!/usr/bin/env python3
"""Re-freeze existing τ2 rows with the explicit grouped-input contract.

This is metadata-only: visible prompts and normalized hashes are unchanged.
Raw official environment trials remain in the collector corpus, while the
trainer aggregates them by router-visible normalized prompt.
"""
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

from theriad.routing.study import (  # noqa: E402
    REPEATED_TRIAL_GROUP_CONTRACT,
    audit_corpus,
    corpus_checksum,
    load_jsonl,
    write_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = list(load_jsonl(args.input))
        before_hashes = {row.prompt_id: row.normalized_hash for row in rows}
        changed = 0
        for index, row in enumerate(rows):
            if row.source != "tau2-bench" or row.task_format != "agent_state":
                continue
            grader = dict(row.grader)
            grader["repeated_trial_contract"] = REPEATED_TRIAL_GROUP_CONTRACT
            grader["router_input_group"] = row.normalized_hash
            grader["effective_sample_unit"] = "normalized_prompt"
            rows[index] = type(row).from_dict(
                {**row.to_dict(), "grader": grader}, require_frozen_hash=True
            )
            changed += 1
        after_hashes = {row.prompt_id: row.normalized_hash for row in rows}
        if before_hashes != after_hashes:
            raise ValueError("metadata-only re-freeze changed a visible prompt hash")
        audit = audit_corpus(rows)
        if not audit.valid:
            raise ValueError(f"re-frozen corpus failed audit: {audit.to_dict()}")
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            groups[row.normalized_hash].append(row)
        duplicate_groups = {
            key: sorted(row.prompt_id for row in values)
            for key, values in groups.items()
            if len(values) > 1
        }
        effective_by_split = Counter(
            (row.split, row.normalized_hash) for row in rows
        )
        effective_counts = Counter(split for split, _ in effective_by_split)
        write_jsonl(rows, args.output)
        report = {
            "schema_version": 1,
            "status": "passed",
            "input_path": str(args.input),
            "output_path": str(args.output),
            "input_sha256": _sha256(args.input),
            "output_sha256": _sha256(args.output),
            "corpus_checksum": corpus_checksum(rows),
            "visible_prompt_text_changed": False,
            "normalized_hashes_preserved": True,
            "rows_rebound": changed,
            "repeated_trial_contract": REPEATED_TRIAL_GROUP_CONTRACT,
            "duplicate_group_count": len(duplicate_groups),
            "raw_trial_count": len(rows),
            "effective_router_input_count": len(effective_by_split),
            "effective_router_input_count_by_split": dict(sorted(effective_counts.items())),
            "duplicate_groups": duplicate_groups,
            "training_policy": {
                "aggregate_outcomes_by_normalized_prompt": True,
                "bootstrap_by_router_input_group": True,
                "hidden_state_added_to_prompt": False,
            },
            "audit": audit.to_dict(),
            "provider_calls": 0,
            "provider_spend_usd": 0.0,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"router-v2 repeated-trial re-freeze stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
