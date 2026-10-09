#!/usr/bin/env python3
"""Evaluate difficulty/disagreement/specialization ablations on public anchors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.anchor_evaluation import (  # noqa: E402
    ABLATION_FEATURES,
    compare_ablation_strategies,
    compare_empirical_vs_irt,
    equal_difficulty_specialization_slice,
    evaluate_selection,
)
from theriad.routing.public_anchors import (  # noqa: E402
    CapabilityProfile,
    assign_grouped_anchor_splits,
    load_anchor_jsonl,
    write_anchor_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id")
    parser.add_argument("--eval-split", choices=("validation", "test", "ood"), default="test")
    parser.add_argument("--quality-threshold", type=float, default=0.8)
    parser.add_argument("--costs", type=Path, required=True, help="JSON object mapping model IDs to costs")
    parser.add_argument("--capability-scores", type=Path)
    parser.add_argument("--capability-source", default="caller-supplied-train-only")
    parser.add_argument("--capability-oof", action="store_true")
    parser.add_argument("--ood-family", action="append", default=[])
    parser.add_argument("--assign-splits", action="store_true")
    parser.add_argument("--write-assigned-anchors", type=Path)
    args = parser.parse_args()

    anchors = load_anchor_jsonl(args.anchors)
    protocols = sorted({observation.protocol_id for anchor in anchors for observation in anchor.observations})
    if len(protocols) > 1 and not args.protocol_id:
        raise SystemExit("--protocol-id is required when anchors contain multiple protocols")
    protocol_id = args.protocol_id or (protocols[0] if protocols else None)
    if args.assign_splits or any(not anchor.split for anchor in anchors):
        anchors = assign_grouped_anchor_splits(
            anchors,
            ood_benchmark_families=args.ood_family,
        )
        if args.write_assigned_anchors:
            write_anchor_jsonl(anchors, args.write_assigned_anchors)
    train = tuple(anchor for anchor in anchors if anchor.split == "train")
    evaluation = tuple(anchor for anchor in anchors if anchor.split == args.eval_split)
    costs = json.loads(args.costs.read_text(encoding="utf-8"))
    if not isinstance(costs, dict) or any(float(value) < 0 for value in costs.values()):
        raise SystemExit("--costs must be a JSON object of non-negative numbers")
    costs = {str(model): float(value) for model, value in costs.items()}

    profile = None
    if args.capability_scores:
        scores = json.loads(args.capability_scores.read_text(encoding="utf-8"))
        if not isinstance(scores, dict):
            raise SystemExit("--capability-scores must contain a JSON object")
        profile = CapabilityProfile.from_scores(
            scores,
            estimated_from=args.capability_source,
            out_of_fold=args.capability_oof,
        )

    result = {
        "anchor_path": str(args.anchors),
        "protocol_id": protocol_id,
        "train_count": len(train),
        "evaluation_count": len(evaluation),
        "eval_split": args.eval_split,
        "ablation_features": {name: list(features) for name, features in ABLATION_FEATURES.items()},
        "strategies": {},
        "paired_ablation_differences": None,
        "equal_difficulty_specialization": equal_difficulty_specialization_slice(
            train,
            evaluation,
            costs=costs,
            capability_profile=profile,
            protocol_id=protocol_id,
        ),
        "irt_comparison": compare_empirical_vs_irt(
            anchors,
            protocol_id=protocol_id,
        ),
    }
    if not train or not evaluation:
        result["status"] = "missing_train_or_evaluation_split"
    else:
        result["status"] = "evaluated"
        result["paired_ablation_differences"] = compare_ablation_strategies(
            train,
            evaluation,
            costs=costs,
            quality_threshold=args.quality_threshold,
            protocol_id=protocol_id,
            capability_profile=profile,
        )
        for strategy in ABLATION_FEATURES:
            curves = []
            for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
                curves.append(
                    evaluate_selection(
                        train,
                        evaluation,
                        strategy=strategy,
                        costs=costs,
                        quality_threshold=threshold,
                        protocol_id=protocol_id,
                        capability_profile=profile,
                    )
                )
            result["strategies"][strategy] = {
                "selected_threshold": evaluate_selection(
                    train,
                    evaluation,
                    strategy=strategy,
                    costs=costs,
                    quality_threshold=args.quality_threshold,
                    protocol_id=protocol_id,
                    capability_profile=profile,
                ),
                "quality_cost_curve": curves,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
