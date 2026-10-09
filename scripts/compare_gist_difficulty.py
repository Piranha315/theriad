#!/usr/bin/env python3
"""Compare frozen and fine-tuned GIST difficulty models on Theriad's grid.

This is an offline analysis.  It loads both local model artifacts, scores the
200 prompts in ``eval_corpus.jsonl``, calibrates each scalar to the grid's Rasch
difficulty with five-fold out-of-fold affine fits, and replays the existing
quality-threshold routing simulation.  It never calls an LLM provider.

The surface predictor's stored out-of-fold predictions remain the incumbent.
The GIST predictors receive exactly the same downstream router, candidate pool,
cost assumptions, tau sweep, and expected-value oracle, so differences isolate
the prompt-difficulty signal rather than a separately tuned routing policy.

Run:

    .venv/bin/python scripts/compare_gist_difficulty.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sim_threshold import TAUS, VALUES, build_router, pool  # noqa: E402
from scripts.train_gist_difficulty import (  # noqa: E402
    choose_device,
    model_kwargs_for_device,
)
from theriad.eval import (  # noqa: E402
    ExpectedValueOracle,
    OutcomeGrid,
    derive_strata,
    fit_rasch,
    load_corpus,
)
from theriad.eval.oracle import Action  # noqa: E402
from theriad.eval.regret import evaluate_router  # noqa: E402
from theriad.models.unified import blended_cost  # noqa: E402

DEFAULT_ARTIFACTS = {
    "gist_frozen": ROOT / "artifacts" / "gist-difficulty-head-only",
    "gist_finetuned": ROOT / "artifacts" / "gist-difficulty",
}
DEFAULT_OUTPUT = ROOT / "artifacts" / "gist-theriad-comparison.json"
FOLDS = 5
SEED = 0


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, without requiring SciPy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    rp, rt = rankdata(prediction), rankdata(target)
    rp -= rp.mean()
    rt -= rt.mean()
    denom = math.sqrt(float(np.sum(rp**2) * np.sum(rt**2)))
    return {
        "count": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(np.sum(rp * rt) / denom) if denom else 0.0,
    }


def make_folds(n: int, *, folds: int = FOLDS, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(n), folds))


def oof_mean(target: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """No-signal baseline fitted only on the other folds."""

    prediction = np.zeros(len(target), dtype=np.float64)
    all_indices = np.arange(len(target))
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        prediction[test] = float(target[train].mean())
    return prediction


def oof_affine_calibration(
    raw: np.ndarray,
    target: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Map a raw score to Rasch logits without fitting on its held-out label."""

    prediction = np.zeros(len(target), dtype=np.float64)
    parameters: list[dict[str, float]] = []
    all_indices = np.arange(len(target))
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        design = np.column_stack([np.ones(len(train)), raw[train]])
        intercept, slope = np.linalg.lstsq(design, target[train], rcond=None)[0]
        # Production calibration needs finite support.  Bounds are learned from
        # the training fold only, so clipping does not inspect held-out labels.
        low, high = float(target[train].min()), float(target[train].max())
        prediction[test] = np.clip(intercept + slope * raw[test], low, high)
        parameters.append(
            {
                "fold": fold_index,
                "intercept": float(intercept),
                "slope": float(slope),
                "train_min": low,
                "train_max": high,
            }
        )
    return prediction, parameters


def score_artifact(
    artifact: Path,
    prompts: list[str],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Load a fully local GIST artifact and return its sigmoid scalar."""

    import torch
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    encoder_path = artifact / "encoder"
    head_path = artifact / "difficulty_head.safetensors"
    if not encoder_path.is_dir() or not head_path.is_file():
        raise FileNotFoundError(f"incomplete GIST artifact: {artifact}")

    model = SentenceTransformer(
        str(encoder_path),
        device=device,
        trust_remote_code=False,
        local_files_only=True,
        model_kwargs=model_kwargs_for_device(device),
    )
    model.eval()
    embeddings = model.encode(
        prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    head = load_file(str(head_path), device="cpu")
    weight = head["weight"].numpy().reshape(-1)
    bias = float(head["bias"].numpy().reshape(-1)[0])
    if embeddings.shape[1] != len(weight):
        raise ValueError(
            f"embedding/head mismatch for {artifact}: "
            f"{embeddings.shape[1]} != {len(weight)}"
        )
    logits = np.asarray(embeddings, dtype=np.float64) @ weight.astype(np.float64) + bias
    # Stable sigmoid; GIST heads are normally far from overflow, but an audit
    # script should not silently emit inf if a future artifact is malformed.
    positive = logits >= 0
    scores = np.empty_like(logits)
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    del model
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return scores


def training_metrics(artifact: Path) -> dict[str, Any]:
    manifest = json.loads((artifact / "training_manifest.json").read_text())
    return {
        "artifact": str(artifact.relative_to(ROOT)),
        "metrics": manifest["metrics"],
        "parameter_counts": manifest["parameter_counts"],
        "training_arguments": manifest["training_arguments"],
    }


def routing_sweep(
    *,
    items,
    grid,
    strata,
    ability: dict[str, float],
    difficulty_by_id: dict[str, float],
    candidates,
    costs: dict[str, float],
) -> dict[str, Any]:
    mean_difficulty = float(np.mean(list(difficulty_by_id.values())))

    def difficulty_of(item):
        return difficulty_by_id.get(item.id, mean_difficulty)

    rows: dict[str, Any] = {}
    for tau in TAUS:
        router = build_router(candidates, ability, difficulty_of, tau)
        selections: dict[str, int] = {}
        for item in items:
            model = router(item).model
            selections[model] = selections.get(model, 0) + 1
        reports = {}
        for value in VALUES:
            oracle = ExpectedValueOracle(
                grid,
                costs,
                value_correct=value,
                strata=strata,
            )
            reports[str(value)] = evaluate_router(
                router,
                items,
                oracle,
                costs,
                strata=strata,
            ).summary()
        rows[str(tau)] = {"selections": selections, "reports": reports}
    return rows


def fixed_baseline_sweep(*, items, grid, strata, costs) -> dict[str, Any]:
    model = "nvidia/nemotron-3-super-120b-a12b:free"

    def router(item):
        return Action(model=model, n_samples=1)

    reports = {}
    for value in VALUES:
        oracle = ExpectedValueOracle(grid, costs, value_correct=value, strata=strata)
        reports[str(value)] = evaluate_router(
            router, items, oracle, costs, strata=strata
        ).summary()
    return {"model": model, "reports": reports}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = choose_device(torch, args.device)
    items = load_corpus(ROOT / "eval_corpus.jsonl")
    grid = OutcomeGrid.load(ROOT / "eval_grid.json")
    fit = fit_rasch(grid)
    fitted_surface = json.loads((ROOT / "difficulty_fit.json").read_text())

    eligible = [item for item in items if item.id in fit.difficulty]
    if len(eligible) != 200:
        raise ValueError(f"expected 200 scored prompts, found {len(eligible)}")
    ids = [item.id for item in eligible]
    if set(ids) != set(fitted_surface["oof_prediction"]):
        raise ValueError("surface predictor IDs do not match the current corpus")

    target = np.asarray([fit.difficulty[item_id] for item_id in ids], dtype=np.float64)
    folds = make_folds(len(eligible))
    raw_scores: dict[str, np.ndarray] = {}
    calibrations: dict[str, list[dict[str, float]]] = {}
    predictions: dict[str, np.ndarray] = {
        "blind_oof_mean": oof_mean(target, folds),
        "surface_ridge": np.asarray(
            [fitted_surface["oof_prediction"][item_id] for item_id in ids],
            dtype=np.float64,
        ),
        "perfect_difficulty_ceiling": target.copy(),
    }

    for name, artifact in DEFAULT_ARTIFACTS.items():
        print(f"scoring {name} on {len(eligible)} prompts ({device})", flush=True)
        raw = score_artifact(
            artifact,
            [item.prompt for item in eligible],
            device=device,
            batch_size=args.batch_size,
        )
        raw_scores[name] = raw
        predictions[name], calibrations[name] = oof_affine_calibration(raw, target, folds)

    predictors: dict[str, Any] = {}
    for name, prediction in predictions.items():
        entry: dict[str, Any] = {
            "theriad_oof_metrics": regression_metrics(prediction, target),
            "prediction_by_item": dict(zip(ids, map(float, prediction), strict=True)),
        }
        if name in raw_scores:
            # Raw RMSE is meaningless because [0,1] source quantiles and Rasch
            # logits use different scales.  Rank correlation is comparable.
            entry["raw_score_spearman"] = regression_metrics(raw_scores[name], target)[
                "spearman"
            ]
            entry["raw_score_by_item"] = dict(
                zip(ids, map(float, raw_scores[name]), strict=True)
            )
            entry["calibration"] = calibrations[name]
            entry["external_training_evaluation"] = training_metrics(
                DEFAULT_ARTIFACTS[name]
            )
        predictors[name] = entry

    candidates = pool()
    missing_abilities = [c.model for c in candidates if c.model not in fit.ability]
    if missing_abilities:
        raise ValueError(f"candidate models missing Rasch ability: {missing_abilities}")
    costs = {
        candidate.model: blended_cost(candidate) / 1_000_000 * 500
        for candidate in candidates
        if blended_cost(candidate) != float("inf")
    }
    strata = derive_strata(grid, eligible)
    mean_difficulty = float(target.mean())
    routing_inputs = {
        "blind_control": {item_id: mean_difficulty for item_id in ids},
        "surface_ridge": predictors["surface_ridge"]["prediction_by_item"],
        "gist_frozen": predictors["gist_frozen"]["prediction_by_item"],
        "gist_finetuned": predictors["gist_finetuned"]["prediction_by_item"],
        "perfect_difficulty_ceiling": predictors["perfect_difficulty_ceiling"][
            "prediction_by_item"
        ],
    }
    routing = {
        name: routing_sweep(
            items=eligible,
            grid=grid,
            strata=strata,
            ability=fit.ability,
            difficulty_by_id=values,
            candidates=candidates,
            costs=costs,
        )
        for name, values in routing_inputs.items()
    }

    external_frozen = predictors["gist_frozen"]["external_training_evaluation"][
        "metrics"
    ]["test"]
    external_finetuned = predictors["gist_finetuned"][
        "external_training_evaluation"
    ]["metrics"]["test"]
    internal = {
        name: predictors[name]["theriad_oof_metrics"]
        for name in (
            "blind_oof_mean",
            "surface_ridge",
            "gist_frozen",
            "gist_finetuned",
        )
    }
    routing_at_tau_095 = {
        name: {
            str(value): routing[name]["0.95"]["reports"][str(value)][
                "regret_per_1k_requests_usd"
            ]
            for value in VALUES
        }
        for name in routing_inputs
    }
    headline = {
        "external_test": {
            "frozen": external_frozen,
            "finetuned": external_finetuned,
            "finetuned_rmse_change_pct": (
                (external_finetuned["rmse"] - external_frozen["rmse"])
                / external_frozen["rmse"]
                * 100
            ),
            "finetuned_spearman_change_pct": (
                (external_finetuned["spearman"] - external_frozen["spearman"])
                / external_frozen["spearman"]
                * 100
            ),
        },
        "theriad_oof": internal,
        "routing_regret_per_1k_at_tau_0_95": routing_at_tau_095,
        "promotion_gates": {
            "finetuned_beats_frozen_external_test_rmse": (
                external_finetuned["rmse"] < external_frozen["rmse"]
            ),
            "finetuned_beats_blind_theriad_rmse": (
                internal["gist_finetuned"]["rmse"]
                < internal["blind_oof_mean"]["rmse"]
            ),
            "finetuned_beats_surface_theriad_rmse": (
                internal["gist_finetuned"]["rmse"]
                < internal["surface_ridge"]["rmse"]
            ),
            "recommend_runtime_promotion": False,
        },
    }

    output = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provider_calls": 0,
        "provider_cost_usd": 0.0,
        "method": {
            "n_items": len(eligible),
            "folds": FOLDS,
            "seed": SEED,
            "calibration": "five-fold OOF affine raw GIST score to Rasch difficulty",
            "routing": "cheapest candidate with sigmoid(ability-difficulty) >= tau",
            "tau_values": TAUS,
            "value_correct_usd": VALUES,
            "cost_assumption": "500 blended tokens per request, matching sim_threshold.py",
            "candidate_models": [candidate.model for candidate in candidates],
        },
        "grid": {
            "cells": len(grid),
            "trials": grid.total_trials,
            "historical_provider_cost_usd": grid.total_cost_usd,
        },
        "headline": headline,
        "predictors": predictors,
        "routing": routing,
        "fixed_no_escalation_baseline": fixed_baseline_sweep(
            items=eligible,
            grid=grid,
            strata=strata,
            costs=costs,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print("\nTheriad held-out difficulty prediction")
    print(f"{'predictor':30}{'MAE':>10}{'RMSE':>10}{'rho':>10}")
    for name in (
        "blind_oof_mean",
        "surface_ridge",
        "gist_frozen",
        "gist_finetuned",
        "perfect_difficulty_ceiling",
    ):
        metric = predictors[name]["theriad_oof_metrics"]
        print(
            f"{name:30}{metric['mae']:>10.3f}{metric['rmse']:>10.3f}"
            f"{metric['spearman']:>10.3f}"
        )

    print("\nRouting at tau=0.95 (the incumbent surface sweep's prior headline)")
    print(f"{'predictor':30}" + "".join(f"{f'V={v}':>12}" for v in VALUES))
    for name in routing_inputs:
        row = routing[name]["0.95"]["reports"]
        values = [row[str(v)]["regret_per_1k_requests_usd"] for v in VALUES]
        print(f"{name:30}" + "".join(f"{value:>12.3f}" for value in values))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
