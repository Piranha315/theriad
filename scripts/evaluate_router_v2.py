#!/usr/bin/env python3
"""Evaluate a frozen Router V2 artifact on validation or untouched test JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.eval.corpus import EvalItem  # noqa: E402
from theriad.eval.routers import theriad_router  # noqa: E402
from theriad.models.tiers import ModelCandidate  # noqa: E402
from theriad.routing.v2 import RouterV2  # noqa: E402
from theriad.routing.v2_training import (  # noqa: E402
    StudyRow,
    artifact_candidates,
    assert_study_row_contract,
    bootstrap_mean_ci,
    brier_score,
    expected_calibration_error,
    log_loss,
)


def _summary(
    successes: list[float], costs: list[float], latencies: list[float] | None = None
) -> dict:
    if not successes:
        return {"count": 0}
    success_ci = bootstrap_mean_ci(successes)
    cost_ci = bootstrap_mean_ci(costs)
    success_total = sum(successes)
    return {
        "count": len(successes),
        "success_rate": float(np.mean(successes)),
        "success_rate_95_ci": list(success_ci),
        "mean_cost_usd": float(np.mean(costs)),
        "mean_cost_95_ci": list(cost_ci),
        "cost_per_success_usd": float(sum(costs) / success_total) if success_total else None,
        "p95_latency_ms": float(np.quantile(latencies, 0.95)) if latencies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nemotron-paid-input", type=float, default=0.0)
    parser.add_argument("--nemotron-paid-output", type=float, default=0.0)
    args = parser.parse_args()

    rows: list[StudyRow] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(StudyRow.from_dict(json.loads(line)))
                except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"{args.input}:{line_number}: {exc}") from exc
    try:
        assert_study_row_contract(rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    evaluation = [row for row in rows if row.split == args.split]
    if not evaluation:
        raise SystemExit(f"no {args.split} rows")

    router = RouterV2.from_path(args.artifact)
    raw_artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    price_snapshot = raw_artifact.get("selection_price_snapshot")
    frozen_candidates = (
        tuple(ModelCandidate(tier="study", **item) for item in price_snapshot)
        if price_snapshot is not None else None
    )
    candidates = artifact_candidates(router.artifact, frozen_candidates)
    successes: list[float] = []
    costs: list[float] = []
    paid_costs: list[float] = []
    free_costs: list[float] = []
    latencies: list[float] = []
    probabilities: list[float] = []
    quality_regret: list[float] = []
    cost_regret: list[float] = []
    slices: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"successes": [], "costs": [], "latencies": []}
    )
    routes: dict[str, int] = defaultdict(int)
    legacy_route = theriad_router()
    legacy_successes: list[float] = []
    legacy_costs: list[float] = []
    legacy_latencies: list[float] = []
    legacy_slices: dict[str, list[float]] = defaultdict(list)
    legacy_missing: dict[str, int] = defaultdict(int)
    for row in evaluation:
        decision = router.predict(row.prompt, candidates, lane=row.lane)
        outcome = row.outcomes.get(decision.selected_model)
        if outcome is None:
            raise SystemExit(f"selected model {decision.selected_model} has no frozen outcome")
        prediction = next(
            candidate for candidate in decision.candidate_scores if candidate.model == decision.selected_model
        )
        success = outcome.success
        cost = outcome.cost_usd
        latency = outcome.latency_ms
        paid_cost = cost
        free_cost = 0.0 if "nemotron-3-super" in decision.selected_model else cost
        if "nemotron-3-super" in decision.selected_model and (
            args.nemotron_paid_input or args.nemotron_paid_output
        ):
            input_tokens = max(1.0, len(row.prompt.split()) * 1.3)
            paid_cost = (
                input_tokens * args.nemotron_paid_input
                + outcome.output_tokens * args.nemotron_paid_output
            ) / 1_000_000.0
        successes.append(success)
        costs.append(cost)
        paid_costs.append(paid_cost)
        free_costs.append(free_cost)
        latencies.append(latency)
        probabilities.append(prediction.p_success)
        best_success = max(item.success for item in row.outcomes.values())
        best_success_models = [
            item for item in row.outcomes.values() if item.success == best_success
        ]
        oracle_cost = min(item.cost_usd for item in best_success_models)
        quality_regret.append(best_success - success)
        cost_regret.append(max(0.0, cost - oracle_cost))
        routes[decision.selected_model] += 1
        previous_mode = os.environ.get("THERIAD_ROUTER_V2_MODE")
        os.environ["THERIAD_ROUTER_V2_MODE"] = "off"
        try:
            legacy_action = legacy_route(
                EvalItem(
                    id=f"study-{len(successes)}",
                    category=row.domain,
                    prompt=row.prompt,
                    source=row.source,
                )
            )
        finally:
            if previous_mode is None:
                os.environ.pop("THERIAD_ROUTER_V2_MODE", None)
            else:
                os.environ["THERIAD_ROUTER_V2_MODE"] = previous_mode
        legacy_outcome = row.outcomes.get(legacy_action.model)
        if legacy_outcome is None:
            legacy_missing[legacy_action.model] += 1
        else:
            legacy_successes.append(legacy_outcome.success)
            legacy_costs.append(legacy_outcome.cost_usd)
            legacy_latencies.append(legacy_outcome.latency_ms)
            legacy_keys = [f"lane:{row.lane}", f"source:{row.source}", f"domain:{row.domain}"]
            if "short-hard" in row.tags:
                legacy_keys.append("short-hard")
            for key in legacy_keys:
                legacy_slices[key].append(legacy_outcome.success)
        keys = [f"lane:{row.lane}", f"source:{row.source}", f"domain:{row.domain}"]
        if len(row.prompt.split()) < 100:
            keys.append("length:short")
        else:
            keys.append("length:long")
        if "short-hard" in row.tags:
            keys.append("short-hard")
        if "long-easy" in row.tags:
            keys.append("long-easy")
        if "ood" in row.tags:
            keys.append("ood")
        for key in keys:
            slices[key]["successes"].append(success)
            slices[key]["costs"].append(cost)
            slices[key]["latencies"].append(latency)

    current_summary = _summary(legacy_successes, legacy_costs, legacy_latencies)
    current_summary["coverage_complete"] = not legacy_missing
    current_summary["missing_routes"] = dict(sorted(legacy_missing.items()))
    new_summary = _summary(successes, costs, latencies)
    free_summary = _summary(successes, free_costs, latencies)
    paid_summary = _summary(successes, paid_costs, latencies)
    non_dominated = None
    if not legacy_missing and legacy_successes:
        legacy_quality = float(np.mean(legacy_successes))
        legacy_cost = float(np.mean(legacy_costs))
        new_quality = float(np.mean(successes))
        new_cost = float(np.mean(costs))
        non_dominated = not (
            legacy_quality >= new_quality
            and legacy_cost <= new_cost
            and (legacy_quality > new_quality or legacy_cost < new_cost)
        )
    report = {
        "primary_evidence": "observed_outcomes_of_runtime_selected_models",
        "candidate_set_source": "frozen_training_prices" if price_snapshot is not None else "runtime_registry",
        "oracle_metrics_are_hindsight_diagnostics": True,
        "router_version": router.artifact.router_version,
        "split": args.split,
        "policy": new_summary,
        "current_router": current_summary,
        "non_dominated_vs_current": non_dominated,
        "free_nemotron_sensitivity": free_summary,
        "paid_nemotron_sensitivity": paid_summary,
        "calibration": {
            "brier": brier_score(probabilities, successes),
            "log_loss": log_loss(probabilities, successes),
            "ece": expected_calibration_error(probabilities, successes),
        },
        "promotion_gates": {
            "ece_at_most_0_05": expected_calibration_error(probabilities, successes) <= 0.05,
            "non_dominated_vs_current": non_dominated,
        },
        "mean_oracle_success_regret": float(np.mean(quality_regret)),
        "mean_oracle_cost_regret_usd": float(np.mean(cost_regret)),
        "routes": dict(sorted(routes.items())),
        "slices": {
            key: _summary(values["successes"], values["costs"], values["latencies"])
            for key, values in sorted(slices.items())
        },
    }
    slice_deltas = {
        key: float(np.mean(values["successes"]) - np.mean(legacy_slices[key]))
        for key, values in slices.items()
        if legacy_slices.get(key)
    }
    report["promotion_gates"].update(
        {
            "major_slice_loss_at_most_3pp": all(
                delta >= -0.03 for delta in slice_deltas.values()
            ),
            "short_hard_improves": slice_deltas.get("short-hard", 0.0) > 0.0,
            "short_hard_delta": slice_deltas.get("short-hard"),
            "p95_latency_available": new_summary.get("p95_latency_ms") is not None,
        }
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
