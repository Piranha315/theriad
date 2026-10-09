#!/usr/bin/env python3
"""Convert resumable collector state into trainable Router V2 JSONL.

The command refuses to emit a promoted training input when any required model
outcome is absent, pending a tool-state signal, pending a judge, truncated, or
errored.  Partial state remains useful for resuming collection, not for fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import (  # noqa: E402
    corpus_checksum,
    load_jsonl,
    study_call_scope,
    validate_corpus,
)
from theriad.routing.study_graders import (  # noqa: E402
    validate_pairwise_contract,
    validate_recorded_grade,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        prompts = load_jsonl(args.corpus)
        audit = validate_corpus(prompts)
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if state.get("corpus_hash"):
            from theriad.routing.study_collector import corpus_fingerprint

            if state["corpus_hash"] != corpus_fingerprint(prompts):
                raise ValueError("collector state belongs to a different corpus")
        state_rows = state.get("rows")
        if not isinstance(state_rows, dict):
            raise ValueError("collector state rows must be an object")

        scope = study_call_scope(prompts)
        required_by_prompt: dict[str, set[str]] = {}
        for model, rows in scope.items():
            for row in rows:
                required_by_prompt.setdefault(row.prompt_id, set()).add(model)
        finalized: list[dict] = []
        missing: list[dict] = []
        excluded_pending_grading: list[dict] = []
        for prompt in prompts:
            state_row = state_rows.get(prompt.prompt_id, {})
            outcomes = state_row.get("outcomes") or {}
            required = required_by_prompt[prompt.prompt_id]
            if prompt.label_type == "pairwise":
                validate_pairwise_contract(prompt.grader)
                missing_models = sorted(model for model in required if model not in outcomes)
                pending_models = sorted(
                    model
                    for model in required
                    if (state_row.get("runs") or {}).get(model, {}).get("status")
                    in {"pending_judge", "pending", "pending_external_state", "missing"}
                )
                if missing_models or pending_models:
                    excluded_pending_grading.append(
                        {
                            "prompt_id": prompt.prompt_id,
                            "split": prompt.split,
                            "reason": "pairwise_judging_not_complete",
                            "missing_models": missing_models,
                            "pending_models": pending_models,
                        }
                    )
                    continue
            missing_models = sorted(model for model in required if model not in outcomes)
            if missing_models:
                missing.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "split": prompt.split,
                        "missing_models": missing_models,
                        "statuses": {
                            model: (state_row.get("runs") or {}).get(model, {}).get("status", "missing")
                            for model in missing_models
                        },
                    }
                )
                continue
            invalid_grades: list[dict[str, str]] = []
            for model in sorted(required):
                run = (state_row.get("runs") or {}).get(model) or {}
                grade = run.get("grade")
                try:
                    if prompt.label_type == "tool_state":
                        validate_recorded_grade(prompt.grader, grade)
                    elif prompt.label_type == "pairwise":
                        validate_pairwise_contract(prompt.grader)
                        if not isinstance(grade, dict) or grade.get("method") != "pairwise_judge":
                            raise ValueError("pairwise outcome lacks a recorded judge grade")
                except ValueError as exc:
                    invalid_grades.append({"model": model, "reason": str(exc)})
            if invalid_grades:
                missing.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "split": prompt.split,
                        "missing_models": [],
                        "invalid_grades": invalid_grades,
                        "statuses": {
                            model: (state_row.get("runs") or {}).get(model, {}).get("status", "missing")
                            for model in sorted(required)
                        },
                    }
                )
                continue
            finalized.append(
                {
                    **prompt.to_dict(),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "outcomes": {model: outcomes[model] for model in sorted(required)},
                }
            )
        report = {
            "corpus_audit": audit.to_dict(),
            "corpus_checksum": corpus_checksum(prompts),
            "required_rows": len(prompts),
            "finalized_rows": len(finalized),
            "incomplete_rows": len(missing),
            "excluded_pending_grading_rows": len(excluded_pending_grading),
            "usable_outcome_count": len(finalized),
            "missing_by_model": dict(
                Counter(model for row in missing for model in row["missing_models"])
            ),
            "missing": missing,
            "excluded_pending_grading": excluded_pending_grading,
            "promotable": not missing and not excluded_pending_grading,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if missing:
            print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in finalized:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), **report}, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"router-v2 finalization stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
