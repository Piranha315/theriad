#!/usr/bin/env python3
"""Regrade full saved responses into a separate, versioned audit artifact.

No providers are contacted and no historical grid or transcript is overwritten.
Incomplete transcript coverage is retained as unknown, never filled from legacy
success counts. The output is a diagnostic audit, not a replacement outcome grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theriad.benchmark.types import GRADER_VERSION, GradeResult
from theriad.eval.corpus import load_corpus


def _source_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def build_regrade_audit(corpus_path: Path, grid_path: Path, transcript_path: Path) -> dict:
    """Build an audit with original grades, new grades, and coverage per cell."""
    items = load_corpus(corpus_path)
    prompts = {item.id: item.to_prompt() for item in items}
    if len(prompts) != len(items):
        raise ValueError("Corpus contains duplicate item IDs")
    original = json.loads(grid_path.read_text())
    cells = {}
    for cell in original["cells"]:
        key = (cell["model"], cell["item_id"])
        if key in cells:
            raise ValueError(f"Grid contains duplicate cell: {key}")
        cells[key] = {
            "model": key[0], "item_id": key[1], "original": cell,
            "transcript_attempts": 0, "transcript_truncated": 0,
            "graded_trials": 0, "successes": 0, "ungradable": 0,
            "recorded_cost_usd": 0.0, "unknown_cost_attempts": 0,
        }

    attempts = []
    status_counts: Counter = Counter()
    changed = 0
    with transcript_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["model"], row["item_id"])
            cell = cells.setdefault(key, {
                "model": key[0], "item_id": key[1], "original": None,
                "transcript_attempts": 0, "transcript_truncated": 0,
                "graded_trials": 0, "successes": 0, "ungradable": 0,
                "recorded_cost_usd": 0.0, "unknown_cost_attempts": 0,
            })
            cell["transcript_attempts"] += 1
            cost = row.get("cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                cell["recorded_cost_usd"] += cost
            else:
                cell["unknown_cost_attempts"] += 1
            output = row.get("output")
            if row.get("truncated") or row.get("finish_reason") in {"length", "max_tokens"}:
                grade = GradeResult("ungradable", "truncated_response")
                cell["transcript_truncated"] += 1
            elif not isinstance(output, str):
                grade = GradeResult("ungradable", "missing_full_output")
            elif key[1] not in prompts:
                grade = GradeResult("ungradable", "unknown_corpus_item")
            else:
                grade = prompts[key[1]].grade(output)
            if grade.passed is None:
                cell["ungradable"] += 1
            else:
                cell["graded_trials"] += 1
                cell["successes"] += int(grade.passed)
            status_counts[grade.status] += 1
            previous = row.get("graded_correct")
            grade_changed = isinstance(previous, bool) and grade.passed is not None and previous != grade.passed
            changed += int(grade_changed)
            attempts.append({
                "source_line": line_number,
                "source_row_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "model": key[0], "item_id": key[1],
                "output_sha256": hashlib.sha256(output.encode()).hexdigest() if isinstance(output, str) else None,
                "original_graded_correct": previous,
                "original_grader_version": row.get("grader_version", "legacy-unversioned"),
                "grade": asdict(grade), "grade_changed": grade_changed,
                "recorded_cost_usd": cost,
            })

    for cell in cells.values():
        prior = cell["original"]
        expected = prior.get("trials", 0) + prior.get("truncated", 0) if prior else None
        cell["expected_original_attempts"] = expected
        cell["attempt_count_matches_grid"] = expected is not None and expected == cell["transcript_attempts"]
        cell["regrade_complete"] = bool(
            prior is not None
            and cell["attempt_count_matches_grid"]
            and cell["graded_trials"] == prior.get("trials", 0)
            and cell["transcript_truncated"] == prior.get("truncated", 0)
            and not cell["unknown_cost_attempts"]
            and math.isclose(cell["recorded_cost_usd"], prior.get("cost_usd", 0.0), abs_tol=1e-7)
        )
    ordered_cells = [cells[key] for key in sorted(cells)]
    return {
        "schema_version": "router-regrade-audit-v1",
        "grader_version": GRADER_VERSION,
        "research_only": True,
        "replacement_grid": False,
        "sources": {
            "corpus": _source_identity(corpus_path),
            "grid": _source_identity(grid_path),
            "transcripts": _source_identity(transcript_path),
        },
        "summary": {
            "transcript_attempts": len(attempts), "status_counts": dict(status_counts),
            "changed_scored_grades": changed, "cells": len(cells),
            "complete_cells": sum(cell["regrade_complete"] for cell in ordered_cells),
            "incomplete_cells": sum(not cell["regrade_complete"] for cell in ordered_cells),
            "coverage_complete": all(cell["regrade_complete"] for cell in ordered_cells),
        },
        "limitations": [
            "Aggregate grid counts cannot prove attempt identity; legacy transcripts have no unique attempt IDs.",
            "Missing or truncated responses remain ungradable; this audit does not measure operational success.",
            "No original success count is used to fill missing grades.",
        ],
        "cells": ordered_cells, "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.corpus.resolve(), args.grid.resolve(), args.transcripts.resolve()}:
        parser.error("Output must differ from every source file")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior artifacts")
    report = build_regrade_audit(args.corpus, args.grid, args.transcripts)
    # Exclusive creation refuses concurrent overwrites as well as user mistakes.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
