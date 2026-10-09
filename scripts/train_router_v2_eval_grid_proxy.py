#!/usr/bin/env python3
"""Build a research-only Router V2 proxy from the existing outcome grid.

This script is deliberately separate from ``scripts/train_router_v2.py``.
``eval_grid.json`` has useful success and cost observations, but it is not a
completed Router V2 study: it has sparse coverage, truncations, no latency or
output-token measurements, and no general/code lane labels.  The output is a
diagnostic report and is not a loadable production router artifact.

The proxy uses a chronological vintage split:

* 2020 and 2021: surface-feature success-head fitting
* 2023: threshold selection and validation
* 2024: untouched diagnostic holdout

Only cells with at least one usable trial and no truncation are used.  A
missing selected cell remains missing; it is never imputed as a failure or a
success.  Coverage is therefore reported beside every routing metric.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.difficulty import featurise  # noqa: E402

VINTAGE_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("2020", "2021"),
    "validation": ("2023",),
    "holdout": ("2024",),
}
MIN_TRAIN_CELLS = 20
MIN_VALIDATION_CELLS = 10
MIN_SELECTION_COVERAGE = 0.80
THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 0.99, 100))


@dataclass(frozen=True)
class Item:
    item_id: str
    prompt: str
    vintage: str
    source: str
    category: str

    @property
    def split(self) -> str:
        for name, vintages in VINTAGE_SPLITS.items():
            if self.vintage in vintages:
                return name
        return "unassigned"


@dataclass(frozen=True)
class Cell:
    model: str
    item_id: str
    trials: int
    successes: int
    cost_usd: float
    truncated: int

    @property
    def usable(self) -> bool:
        """Whether the cell can contribute a quality/cost observation."""

        return self.trials > 0 and self.truncated == 0

    @property
    def success_rate(self) -> float:
        if self.trials <= 0:
            raise ValueError("success_rate requested for a zero-trial cell")
        return self.successes / self.trials

    @property
    def cost_per_trial_usd(self) -> float:
        if self.trials <= 0:
            raise ValueError("cost_per_trial requested for a zero-trial cell")
        return self.cost_usd / self.trials


@dataclass(frozen=True)
class Head:
    model: str
    beta: np.ndarray
    train_count: int
    validation_count: int
    train_mean_success: float
    expected_cost_usd: float


def _require_int(raw: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _require_float(raw: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = raw.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{key} must be finite and >= {minimum}")
    return number


def load_inputs(
    grid_path: Path, corpus_path: Path
) -> tuple[list[Item], dict[tuple[str, str], Cell], dict[str, Any]]:
    """Load and validate the corpus/grid join without changing either file."""

    corpus: list[Item] = []
    seen_items: set[str] = set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{corpus_path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{corpus_path}:{line_number}: corpus row must be an object")
            item_id = str(raw.get("id") or "").strip()
            prompt = str(raw.get("prompt") or "").strip()
            vintage = str(raw.get("vintage") or "").strip()
            if not item_id or not prompt or not vintage:
                raise ValueError(
                    f"{corpus_path}:{line_number}: id, prompt, and vintage are required"
                )
            if item_id in seen_items:
                raise ValueError(f"{corpus_path}:{line_number}: duplicate item id {item_id}")
            seen_items.add(item_id)
            corpus.append(
                Item(
                    item_id=item_id,
                    prompt=prompt,
                    vintage=vintage,
                    source=str(raw.get("source") or "unknown"),
                    category=str(raw.get("category") or "unknown"),
                )
            )

    grid_raw = json.loads(grid_path.read_text(encoding="utf-8"))
    if not isinstance(grid_raw, Mapping) or not isinstance(grid_raw.get("cells"), list):
        raise ValueError(f"{grid_path}: expected an object with a cells list")
    items_by_id = {item.item_id: item for item in corpus}
    cells: dict[tuple[str, str], Cell] = {}
    for index, raw in enumerate(grid_raw["cells"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{grid_path}: cells[{index}] must be an object")
        model = str(raw.get("model") or "").strip()
        item_id = str(raw.get("item_id") or "").strip()
        if not model or not item_id:
            raise ValueError(f"{grid_path}: cells[{index}] needs model and item_id")
        if item_id not in items_by_id:
            raise ValueError(f"{grid_path}: cells[{index}] references unknown item {item_id}")
        key = (model, item_id)
        if key in cells:
            raise ValueError(f"{grid_path}: duplicate cell {model}/{item_id}")
        trials = _require_int(raw, "trials")
        successes = _require_int(raw, "successes")
        truncated = _require_int(raw, "truncated")
        if successes > trials:
            raise ValueError(f"{grid_path}: successes exceeds trials for {model}/{item_id}")
        cells[key] = Cell(
            model=model,
            item_id=item_id,
            trials=trials,
            successes=successes,
            cost_usd=_require_float(raw, "cost_usd"),
            truncated=truncated,
        )
    metadata = {
        "grid_version": grid_raw.get("version"),
        "grid_cell_count": len(cells),
        "corpus_item_count": len(corpus),
        "corpus_item_ids": sorted(items_by_id),
    }
    return corpus, cells, metadata


def _logistic_fit(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0) -> np.ndarray:
    """Small deterministic IRLS fit with an unpenalized intercept."""

    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for _ in range(100):
        eta = np.clip(x @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), 1e-6, None)
        z = eta + (y - probability) / weight
        lhs = x.T @ (weight[:, None] * x) + penalty
        rhs = x.T @ (weight * z)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if float(np.max(np.abs(updated - beta))) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _brier(probability: Sequence[float], target: Sequence[float]) -> float:
    return float(np.mean((np.asarray(probability) - np.asarray(target)) ** 2))


def _log_loss(probability: Sequence[float], target: Sequence[float]) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    y = np.asarray(target, dtype=np.float64)
    return float(np.mean(-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)))


def _ece(
    probability: Sequence[float], target: Sequence[float], *, bins: int = 10
) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        upper = p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        mask = (p >= edges[index]) & upper
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean() - y[mask].mean()))
    return float(total)


def _feature_data(
    items: Sequence[Item],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    raw_features = {item.item_id: featurise(item.prompt) for item in items}
    names = sorted({name for features in raw_features.values() for name in features})
    matrix = {
        item_id: np.asarray(
            [float(raw_features[item_id].get(name, 0.0)) for name in names],
            dtype=np.float64,
        )
        for item_id in raw_features
    }
    for item_id, row in matrix.items():
        if row.size != len(names) or not np.isfinite(row).all():
            raise ValueError(f"non-finite or malformed feature row for {item_id}")
    training_ids = [item.item_id for item in items if item.split == "train"]
    training_matrix = np.asarray([matrix[item_id] for item_id in training_ids])
    mean = training_matrix.mean(axis=0)
    stdev = training_matrix.std(axis=0)
    stdev[stdev < 1e-9] = 1.0
    standardized = {
        item_id: (row - mean) / stdev for item_id, row in matrix.items()
    }
    return (
        names,
        standardized,
        {name: float(value) for name, value in zip(names, mean, strict=True)},
        {name: float(value) for name, value in zip(names, stdev, strict=True)},
    )


def _split_items(items: Sequence[Item], split: str) -> list[Item]:
    return [item for item in items if item.split == split]


def _usable_cell(
    cells: Mapping[tuple[str, str], Cell], model: str, item_id: str
) -> Cell | None:
    cell = cells.get((model, item_id))
    return cell if cell is not None and cell.usable else None


def _model_split_profile(
    model: str,
    items: Sequence[Item],
    cells: Mapping[tuple[str, str], Cell],
    split: str,
) -> dict[str, Any]:
    target = _split_items(items, split)
    observed = [cells[(model, item.item_id)] for item in target if (model, item.item_id) in cells]
    usable = [cell for cell in observed if cell.usable]
    trial_count = sum(cell.trials for cell in observed)
    truncation_count = sum(cell.truncated for cell in observed)
    result: dict[str, Any] = {
        "target_items": len(target),
        "recorded_cells": len(observed),
        "usable_cells": len(usable),
        "missing_cells": len(target) - len(observed),
        "truncated_cells": sum(cell.truncated > 0 for cell in observed),
        "trials": trial_count,
        "truncated_calls": truncation_count,
        "usable_cell_coverage": len(usable) / len(target) if target else None,
        "usable_success_rate": (
            sum(cell.successes for cell in usable) / sum(cell.trials for cell in usable)
            if usable
            else None
        ),
        "usable_mean_cost_usd_per_request": (
            float(np.mean([cell.cost_per_trial_usd for cell in usable])) if usable else None
        ),
    }
    return result


def _fit_heads(
    items: Sequence[Item],
    cells: Mapping[tuple[str, str], Cell],
    standardized_features: Mapping[str, np.ndarray],
    expected_costs: dict[str, float],
    *,
    min_train_cells: int,
    min_validation_cells: int,
) -> tuple[dict[str, Head], dict[str, dict[str, Any]]]:
    train = _split_items(items, "train")
    validation = _split_items(items, "validation")
    models = sorted({model for model, _item_id in cells})
    heads: dict[str, Head] = {}
    fit_metadata: dict[str, dict[str, Any]] = {}
    for model in models:
        train_cells = [
            _usable_cell(cells, model, item.item_id)
            for item in train
        ]
        train_pairs = [
            (item, cell)
            for item, cell in zip(train, train_cells, strict=True)
            if cell is not None
        ]
        validation_count = sum(
            _usable_cell(cells, model, item.item_id) is not None for item in validation
        )
        fit_metadata[model] = {
            "train_usable_cells": len(train_pairs),
            "validation_usable_cells": validation_count,
            "fit_status": "not_fit",
            "excluded_reason": None,
        }
        if len(train_pairs) < min_train_cells:
            fit_metadata[model]["excluded_reason"] = "below_minimum_train_coverage"
            continue
        x_train = np.asarray(
            [np.r_[1.0, standardized_features[item.item_id]] for item, _cell in train_pairs],
            dtype=np.float64,
        )
        y_train = np.asarray([cell.success_rate for _item, cell in train_pairs], dtype=np.float64)
        beta = _logistic_fit(x_train, y_train)
        expected_cost = float(np.mean([cell.cost_per_trial_usd for _item, cell in train_pairs]))
        expected_costs[model] = expected_cost
        heads[model] = Head(
            model=model,
            beta=beta,
            train_count=len(train_pairs),
            validation_count=validation_count,
            train_mean_success=float(np.mean(y_train)),
            expected_cost_usd=expected_cost,
        )
        fit_metadata[model].update(
            {
                "fit_status": "fit",
                "excluded_reason": (
                    None
                    if validation_count >= min_validation_cells
                    else "below_minimum_validation_coverage"
                ),
            }
        )
    return heads, fit_metadata


def _prediction_metrics(
    model: str,
    target_items: Sequence[Item],
    cells: Mapping[tuple[str, str], Cell],
    predictions: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    pairs = [
        (predictions[model][item.item_id], cell.success_rate)
        for item in target_items
        for cell in [_usable_cell(cells, model, item.item_id)]
        if cell is not None
        and model in predictions
        and item.item_id in predictions[model]
    ]
    if not pairs:
        return {"status": "insufficient_observations", "n": 0}
    probability, target = zip(*pairs, strict=True)
    return {
        "status": "ok",
        "n": len(pairs),
        "brier": _brier(probability, target),
        "log_loss": _log_loss(probability, target),
        "ece": _ece(probability, target),
        "mean_predicted_success": float(np.mean(probability)),
        "mean_observed_success": float(np.mean(target)),
        "accuracy_at_0_5": float(
            np.mean(
                (np.asarray(probability) >= 0.5)
                == (np.asarray(target) >= 0.5)
            )
        ),
    }


def _pareto_frontier(points: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    ordered = sorted(points, key=lambda point: (point[1], -point[2], point[0]))
    frontier: list[tuple[float, float, float]] = []
    best_quality = -math.inf
    for point in ordered:
        if point[2] > best_quality + 1e-12:
            frontier.append(point)
            best_quality = point[2]
    return frontier


def _automatic_knee(points: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    frontier = _pareto_frontier(points)
    if not frontier:
        raise ValueError("at least one threshold point is required")
    if len(frontier) <= 2:
        return max(frontier, key=lambda point: (point[2] - point[1], -point[1], -point[0]))
    costs = np.asarray([point[1] for point in frontier], dtype=np.float64)
    qualities = np.asarray([point[2] for point in frontier], dtype=np.float64)
    cost_range = max(1e-12, float(costs.max() - costs.min()))
    quality_range = max(1e-12, float(qualities.max() - qualities.min()))
    x = (costs - costs.min()) / cost_range
    y = (qualities - qualities.min()) / quality_range
    x1, y1 = float(x[0]), float(y[0])
    x2, y2 = float(x[-1]), float(y[-1])
    denominator = math.hypot(y2 - y1, x2 - x1) or 1.0
    distances = [
        ((x2 - x1) * float(yi) - (y2 - y1) * float(xi) + y2 * x1 - x2 * y1)
        / denominator
        for xi, yi in zip(x, y, strict=True)
    ]
    best = max(
        range(len(frontier)),
        key=lambda index: (distances[index], -frontier[index][1], -frontier[index][0]),
    )
    return frontier[best]


def _choose_model(
    predictions: Mapping[str, float],
    candidate_models: Sequence[str],
    expected_costs: Mapping[str, float],
    threshold: float,
) -> str:
    clearing = [model for model in candidate_models if predictions[model] >= threshold]
    pool = clearing or list(candidate_models)
    if clearing:
        return min(
            pool,
            key=lambda model: (expected_costs[model], -predictions[model], model),
        )
    return min(
        pool,
        key=lambda model: (-predictions[model], expected_costs[model], model),
    )


def _summarize_choices(
    target_items: Sequence[Item],
    choices: Mapping[str, str],
    cells: Mapping[tuple[str, str], Cell],
    expected_costs: Mapping[str, float],
    *,
    policy_kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_models = list(choices.values())
    scored = [
        (item, choices[item.item_id], _usable_cell(cells, choices[item.item_id], item.item_id))
        for item in target_items
        if item.item_id in choices
        and _usable_cell(cells, choices[item.item_id], item.item_id) is not None
    ]
    selected_counter = Counter(selected_models)
    scored_counter = Counter(model for _item, model, _cell in scored)
    target_count = len(target_items)
    scored_count = len(scored)
    result: dict[str, Any] = {
        "policy_kind": policy_kind,
        "target_items": target_count,
        "selected_items": len(selected_models),
        "scored_items": scored_count,
        "coverage": scored_count / target_count if target_count else None,
        "observed_success_rate_scored": (
            float(np.mean([cell.success_rate for _item, _model, cell in scored]))
            if scored
            else None
        ),
        "observed_mean_cost_usd_per_scored_request": (
            float(np.mean([cell.cost_per_trial_usd for _item, _model, cell in scored]))
            if scored
            else None
        ),
        "observed_total_cost_usd_scored_requests": (
            float(sum(cell.cost_per_trial_usd for _item, _model, cell in scored))
            if scored
            else None
        ),
        "expected_mean_cost_usd_per_decision": (
            float(np.mean([expected_costs[model] for model in selected_models]))
            if selected_models
            else None
        ),
        "model_usage": {
            model: {
                "selected_items": selected_counter[model],
                "selected_share_of_target": (
                    selected_counter[model] / target_count if target_count else None
                ),
                "scored_items": scored_counter[model],
                "scored_share": scored_counter[model] / scored_count if scored_count else None,
            }
            for model in sorted(set(selected_models))
        },
    }
    if extra:
        result.update(extra)
    return result


def _policy_summary(
    target_items: Sequence[Item],
    predictions: Mapping[str, Mapping[str, float]],
    cells: Mapping[tuple[str, str], Cell],
    candidate_models: Sequence[str],
    expected_costs: Mapping[str, float],
    threshold: float,
) -> dict[str, Any]:
    choices = {
        item.item_id: _choose_model(
            {model: predictions[model][item.item_id] for model in candidate_models},
            candidate_models,
            expected_costs,
            threshold,
        )
        for item in target_items
    }
    return _summarize_choices(
        target_items,
        choices,
        cells,
        expected_costs,
        policy_kind="surface_success_threshold",
        extra={"threshold": threshold},
    )


def _fixed_summary(
    target_items: Sequence[Item],
    model: str,
    cells: Mapping[tuple[str, str], Cell],
    expected_costs: Mapping[str, float],
    *,
    policy_kind: str,
) -> dict[str, Any]:
    choices = {item.item_id: model for item in target_items}
    return _summarize_choices(
        target_items,
        choices,
        cells,
        expected_costs,
        policy_kind=policy_kind,
        extra={"model": model},
    )


def _oracle_summary(
    target_items: Sequence[Item],
    cells: Mapping[tuple[str, str], Cell],
    expected_costs: Mapping[str, float],
    models: Sequence[str],
) -> dict[str, Any]:
    choices: dict[str, str] = {}
    for item in target_items:
        observed = [
            cell
            for model in models
            for cell in [_usable_cell(cells, model, item.item_id)]
            if cell is not None
        ]
        if observed:
            best_success = max(cell.success_rate for cell in observed)
            best = [cell for cell in observed if cell.success_rate == best_success]
            choices[item.item_id] = min(
                best,
                key=lambda cell: (cell.cost_per_trial_usd, cell.model),
            ).model
    return _summarize_choices(
        target_items,
        choices,
        cells,
        expected_costs,
        policy_kind="itemwise_observed_quality_oracle",
        extra={
            "interpretation": (
                "Optimistic upper bound using the realized target-cell outcome; "
                "not available to a live router."
            )
        },
    )


def _threshold_sweep(
    target_items: Sequence[Item],
    predictions: Mapping[str, Mapping[str, float]],
    cells: Mapping[tuple[str, str], Cell],
    candidate_models: Sequence[str],
    expected_costs: Mapping[str, float],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        summary = _policy_summary(
            target_items,
            predictions,
            cells,
            candidate_models,
            expected_costs,
            threshold,
        )
        points.append(
            {
                "threshold": threshold,
                "coverage": summary["coverage"],
                "scored_items": summary["scored_items"],
                "observed_success_rate_scored": summary["observed_success_rate_scored"],
                "observed_mean_cost_usd_per_scored_request": summary[
                    "observed_mean_cost_usd_per_scored_request"
                ],
                "expected_mean_cost_usd_per_decision": summary[
                    "expected_mean_cost_usd_per_decision"
                ],
            }
        )
    return points


def _select_threshold(
    sweep: Sequence[Mapping[str, Any]],
    *,
    min_coverage: float,
) -> dict[str, Any]:
    eligible = [
        point
        for point in sweep
        if point["coverage"] is not None
        and point["coverage"] >= min_coverage
        and point["scored_items"] >= 10
        and point["observed_success_rate_scored"] is not None
        and point["observed_mean_cost_usd_per_scored_request"] is not None
    ]
    if eligible:
        triples = [
            (
                float(point["threshold"]),
                float(point["observed_mean_cost_usd_per_scored_request"]),
                float(point["observed_success_rate_scored"]),
            )
            for point in eligible
        ]
        knee_threshold, knee_cost, knee_success = _automatic_knee(triples)
        selected = min(eligible, key=lambda point: abs(float(point["threshold"]) - knee_threshold))
        return {
            "status": "validation_knee",
            "minimum_coverage": min_coverage,
            "eligible_threshold_count": len(eligible),
            "threshold": float(selected["threshold"]),
            "validation_observed_mean_cost_usd_per_scored_request": knee_cost,
            "validation_observed_success_rate_scored": knee_success,
        }
    selected = max(
        sweep,
        key=lambda point: (
            -1.0 if point["coverage"] is None else float(point["coverage"]),
            -1.0
            if point["observed_success_rate_scored"] is None
            else float(point["observed_success_rate_scored"]),
            -float(point["threshold"]),
        ),
    )
    return {
        "status": "coverage_fallback",
        "minimum_coverage": min_coverage,
        "eligible_threshold_count": 0,
        "threshold": float(selected["threshold"]),
        "validation_observed_mean_cost_usd_per_scored_request": selected[
            "observed_mean_cost_usd_per_scored_request"
        ],
        "validation_observed_success_rate_scored": selected["observed_success_rate_scored"],
    }


def build_report(
    items: Sequence[Item],
    cells: Mapping[tuple[str, str], Cell],
    input_metadata: Mapping[str, Any],
    *,
    grid_path: Path,
    corpus_path: Path,
    min_train_cells: int,
    min_validation_cells: int,
    min_selection_coverage: float,
) -> dict[str, Any]:
    feature_names, standardized, feature_mean, feature_stdev = _feature_data(items)
    expected_costs: dict[str, float] = {}
    heads, fit_metadata = _fit_heads(
        items,
        cells,
        standardized,
        expected_costs,
        min_train_cells=min_train_cells,
        min_validation_cells=min_validation_cells,
    )
    for model in heads:
        fit_metadata[model]["candidate_for_policy"] = (
            heads[model].validation_count >= min_validation_cells
        )
    candidate_models = sorted(
        model for model, head in heads.items() if head.validation_count >= min_validation_cells
    )
    if not candidate_models:
        raise ValueError("no model has enough train and validation observations for proxy policy")

    predictions: dict[str, dict[str, float]] = {
        model: {
            item.item_id: float(
                _sigmoid(
                    np.asarray(
                        [1.0, *standardized[item.item_id]],
                        dtype=np.float64,
                    )
                    @ head.beta
                )
            )
            for item in items
        }
        for model, head in heads.items()
    }

    train = _split_items(items, "train")
    validation = _split_items(items, "validation")
    holdout = _split_items(items, "holdout")
    validation_predictions = {model: predictions[model] for model in heads}
    sweep = _threshold_sweep(
        validation,
        validation_predictions,
        cells,
        candidate_models,
        expected_costs,
    )
    selection = _select_threshold(sweep, min_coverage=min_selection_coverage)
    selected_threshold = float(selection["threshold"])

    cheapest_model = min(
        candidate_models,
        key=lambda model: (expected_costs[model], -heads[model].train_mean_success, model),
    )
    strongest_model = max(
        candidate_models,
        key=lambda model: (heads[model].train_mean_success, -expected_costs[model], model),
    )
    dearest_model = max(
        candidate_models,
        key=lambda model: (expected_costs[model], heads[model].train_mean_success, model),
    )

    def split_report(target: Sequence[Item]) -> dict[str, Any]:
        fixed = {
            "always_cheapest_train_cost": _fixed_summary(
                target,
                cheapest_model,
                cells,
                expected_costs,
                policy_kind="always_cheapest_train_cost",
            ),
            "strongest_fixed_train_success": _fixed_summary(
                target,
                strongest_model,
                cells,
                expected_costs,
                policy_kind="strongest_fixed_train_success",
            ),
            "always_dearest_train_cost": _fixed_summary(
                target,
                dearest_model,
                cells,
                expected_costs,
                policy_kind="always_dearest_train_cost",
            ),
            "itemwise_observed_quality_oracle": _oracle_summary(
                target,
                cells,
                expected_costs,
                candidate_models,
            ),
        }
        return {
            "items": len(target),
            "proxy_policy": _policy_summary(
                target,
                predictions,
                cells,
                candidate_models,
                expected_costs,
                selected_threshold,
            ),
            "baselines": fixed,
        }

    model_profiles = {
        model: {
            "pricing_class_hint": "free_suffix" if model.endswith(":free") else "non_free_or_unclassified",
            "coverage": {
                split: _model_split_profile(model, items, cells, split)
                for split in ("train", "validation", "holdout")
            },
            "fit": fit_metadata.get(
                model,
                {
                    "fit_status": "not_fit",
                    "excluded_reason": "below_minimum_train_coverage",
                },
            ),
        }
        for model in sorted({model for model, _item_id in cells})
    }
    for model, head in heads.items():
        model_profiles[model]["head"] = {
            "intercept": float(head.beta[0]),
            "weights": {
                name: float(value)
                for name, value in zip(feature_names, head.beta[1:], strict=True)
            },
            "train_mean_success": head.train_mean_success,
            "expected_cost_usd_per_request": head.expected_cost_usd,
            "train_usable_cells": head.train_count,
            "validation_usable_cells": head.validation_count,
        }
        model_profiles[model]["prediction_quality"] = {
            "validation": _prediction_metrics(model, validation, cells, predictions),
            "holdout": _prediction_metrics(model, holdout, cells, predictions),
        }

    usable_cells = [cell for cell in cells.values() if cell.usable]
    truncated_cells = [cell for cell in cells.values() if cell.truncated > 0]
    zero_trial_cells = [cell for cell in cells.values() if cell.trials == 0]
    source_models = sorted({model for model, _item_id in cells})
    report = {
        "report_kind": "router_v2_eval_grid_proxy_diagnostic",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_eligible": False,
        "loadable_router_artifact": False,
        "router_v2_enabled": False,
        "runtime_mutations": [],
        "frozen_router_v2_study_test_set_used": False,
        "sources": {
            "grid": str(grid_path.resolve()),
            "corpus": str(corpus_path.resolve()),
            **dict(input_metadata),
            "model_count": len(source_models),
            "models": source_models,
            "vintage_counts": dict(
                sorted(Counter(item.vintage for item in items).items())
            ),
            "source_counts": dict(sorted(Counter(item.source for item in items).items())),
            "category_counts": dict(sorted(Counter(item.category for item in items).items())),
        },
        "split_definition": {
            "train": list(VINTAGE_SPLITS["train"]),
            "validation": list(VINTAGE_SPLITS["validation"]),
            "diagnostic_holdout": list(VINTAGE_SPLITS["holdout"]),
            "holdout_tuning_used": False,
            "note": (
                "The 2024 vintage is an eval-grid diagnostic holdout. It is not "
                "the frozen Router V2 study test split."
            ),
        },
        "data_quality": {
            "grid_cells": len(cells),
            "usable_cells": len(usable_cells),
            "non_usable_cells_excluded_union": len(cells) - len(usable_cells),
            "zero_trial_cells_excluded": len(zero_trial_cells),
            "truncated_cells_excluded": len(truncated_cells),
            "cells_with_trials_and_truncation": sum(
                cell.trials > 0 and cell.truncated > 0 for cell in cells.values()
            ),
            "total_trials": sum(cell.trials for cell in cells.values()),
            "total_truncated_calls": sum(cell.truncated for cell in cells.values()),
            "total_recorded_cost_usd": float(sum(cell.cost_usd for cell in cells.values())),
            "cost_handling": (
                "cost_usd is treated as accumulated cell cost and divided by usable "
                "trials; train means are used for routing decisions."
            ),
            "latency_available": False,
            "output_tokens_available": False,
            "latency_or_output_tokens_fabricated": False,
            "transcripts_used_for_fit": False,
        },
        "method": {
            "representation": "prompt surface features from theriad.routing.difficulty.featurise",
            "model": "one deterministic ridge-logistic success head per model",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "feature_mean_train": feature_mean,
            "feature_stdev_train": feature_stdev,
            "minimum_train_usable_cells": min_train_cells,
            "minimum_validation_usable_cells": min_validation_cells,
            "candidate_models": candidate_models,
            "excluded_from_policy": sorted(set(source_models) - set(candidate_models)),
            "routing_rule": (
                "choose the lowest expected-cost candidate whose predicted success "
                "is at least the threshold; otherwise choose the highest predicted success"
            ),
            "threshold_selection": (
                "validation-only Pareto knee over observed success and observed cost, "
                "requiring the configured minimum scored coverage"
            ),
            "minimum_selection_coverage": min_selection_coverage,
        },
        "models": model_profiles,
        "threshold_selection": {
            **selection,
            "validation_threshold_sweep": sweep,
        },
        "evaluation": {
            "train": split_report(train),
            "validation": split_report(validation),
            "diagnostic_holdout": split_report(holdout),
        },
        "limitations": [
            "This is a diagnostic proxy, not official Router V2 training or a production artifact.",
            "The grid has sparse/asymmetric model coverage; scored routing metrics are conditional on a usable selected cell.",
            "Truncated and zero-trial cells are excluded rather than treated as failures.",
            "The grid has no reliable latency or output-token fields, so this report cannot evaluate latency or token efficiency.",
            "The corpus has no general/code lane labels, so this proxy reports one unified lane.",
            "The itemwise oracle uses realized outcomes and is an optimistic upper bound, not a deployable policy.",
            "No Router V2 environment flag, registry entry, or runtime artifact is changed by this script.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "eval_grid.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval_corpus.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results/router_v2_eval_grid_proxy.json",
    )
    parser.add_argument("--min-train-cells", type=int, default=MIN_TRAIN_CELLS)
    parser.add_argument("--min-validation-cells", type=int, default=MIN_VALIDATION_CELLS)
    parser.add_argument(
        "--min-selection-coverage",
        type=float,
        default=MIN_SELECTION_COVERAGE,
    )
    args = parser.parse_args()
    if args.min_train_cells < 1 or args.min_validation_cells < 1:
        raise SystemExit("minimum cell counts must be positive")
    if not 0.0 <= args.min_selection_coverage <= 1.0:
        raise SystemExit("--min-selection-coverage must be in [0, 1]")
    items, cells, input_metadata = load_inputs(args.grid, args.corpus)
    report = build_report(
        items,
        cells,
        input_metadata,
        grid_path=args.grid,
        corpus_path=args.corpus,
        min_train_cells=args.min_train_cells,
        min_validation_cells=args.min_validation_cells,
        min_selection_coverage=args.min_selection_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected = report["threshold_selection"]
    holdout = report["evaluation"]["diagnostic_holdout"]["proxy_policy"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "production_eligible": report["production_eligible"],
                "models_in_grid": report["sources"]["model_count"],
                "candidate_models": len(report["method"]["candidate_models"]),
                "selected_threshold": selected["threshold"],
                "holdout_scored_items": holdout["scored_items"],
                "holdout_coverage": holdout["coverage"],
                "holdout_observed_success_rate_scored": holdout[
                    "observed_success_rate_scored"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
