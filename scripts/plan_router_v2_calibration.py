#!/usr/bin/env python3
"""Size and dry-run-plan the revised small Theriad calibration study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.calibration_study import estimate_calibration_size  # noqa: E402
from theriad.routing.public_anchors import load_anchor_jsonl  # noqa: E402
from theriad.routing.study import StudyPrompt  # noqa: E402
from theriad.routing.study_collector import (  # noqa: E402
    build_calibration_study_plan,
    write_plan,
)


def _load_calibration_prompts(path: Path) -> tuple[StudyPrompt, ...]:
    rows: list[StudyPrompt] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rows.append(StudyPrompt.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise SystemExit(f"{path}:{line_number}: invalid calibration prompt: {exc}") from exc
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--calibration-corpus", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target-half-width", type=float, default=0.05)
    parser.add_argument("--transfer-correlation", type=float, default=0.0)
    parser.add_argument("--design-effect", type=float, default=1.0)
    parser.add_argument("--minimum-rows-per-domain", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()

    anchors = load_anchor_jsonl(args.anchors)
    sizing = estimate_calibration_size(
        anchors,
        confidence=args.confidence,
        target_half_width=args.target_half_width,
        assumed_transfer_correlation=args.transfer_correlation,
        design_effect=args.design_effect,
        minimum_rows_per_domain=args.minimum_rows_per_domain,
    )
    prompts = _load_calibration_prompts(args.calibration_corpus)
    report = sizing.to_dict()
    report.update(
        {
            "calibration_prompt_count": len(prompts),
            "calibration_models": list(dict.fromkeys(args.models)),
            "calibration_prompt_count_meets_recommendation": len(prompts) >= sizing.recommended_rows,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sizing.status != "sized":
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    plan = build_calibration_study_plan(
        prompts,
        models=tuple(args.models),
        max_output_tokens=args.max_output_tokens,
        max_attempts=args.max_attempts,
    )
    write_plan(plan, args.plan)
    print(json.dumps({"report": report, "plan": plan.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
