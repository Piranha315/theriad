"""Held-out evaluation for public-anchor difficulty/disagreement features."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .public_anchors import (
    AnchorObservation,
    CapabilityProfile,
    PublicAnchor,
    derive_anchor_metrics,
)

ABLATION_FEATURES: dict[str, tuple[str, ...]] = {
    "difficulty": ("empirical_difficulty",),
    "difficulty_disagreement": (
        "empirical_difficulty",
        "model_disagreement",
    ),
    "difficulty_disagreement_specialization": (
        "empirical_difficulty",
        "model_disagreement",
        "specialization_score",
        "prompt_chars",
        "prompt_words",
        "has_code",
        "has_question",
        "domain_one_hot",
        "anchor_neighbors",
    ),
}


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    if isinstance(value, np.ndarray):
        clipped = np.clip(value, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _logistic_fit(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = np.asarray(_sigmoid(x @ beta), dtype=np.float64)
        weight = np.clip(probability * (1.0 - probability), 1e-6, None)
        z = x @ beta + (y - probability) / weight
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


def _metric(anchor: PublicAnchor, name: str, default: float = 0.5) -> float:
    value = anchor.metric_map.get(name)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return default


def _anchor_feature_values(
    anchors: Sequence[PublicAnchor],
    *,
    domains: Sequence[str],
    features: Sequence[str],
    target_model: str | None = None,
    protocol_id: str | None = None,
    capability_profile: CapabilityProfile | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names: list[str] = []
    for feature in features:
        if feature == "domain_one_hot":
            names.extend(f"domain={domain}" for domain in domains)
        elif feature == "anchor_neighbors":
            names.extend(
                (
                    "nearest_anchor_similarity",
                    "neighbor_mean_difficulty",
                    "neighbor_mean_disagreement",
                    "neighbor_specialization_score",
                )
            )
        else:
            names.append(feature)
    values = np.zeros((len(anchors), len(names)), dtype=np.float64)
    for row_index, anchor in enumerate(anchors):
        cursor = 0
        feature_observations = (
            tuple(
                observation
                for observation in anchor.observations
                if observation.model_id != target_model
            )
            if target_model is not None
            else anchor.observations
        )
        metrics = derive_anchor_metrics(
            feature_observations,
            capability_profile=capability_profile,
            protocol_id=protocol_id,
        )
        stored_metrics = anchor.metric_map
        if "anchor_features" in stored_metrics:
            metrics["anchor_features"] = stored_metrics["anchor_features"]
        for feature in features:
            if feature == "domain_one_hot":
                for domain in domains:
                    values[row_index, cursor] = float(anchor.domain == domain)
                    cursor += 1
            elif feature == "anchor_neighbors":
                nested = metrics.get("anchor_features")
                nested = nested if isinstance(nested, Mapping) else {}
                for key in (
                    "nearest_anchor_similarity",
                    "neighbor_mean_difficulty",
                    "neighbor_mean_disagreement",
                    "neighbor_specialization_score",
                ):
                    value = nested.get(key)
                    values[row_index, cursor] = (
                        float(value) if isinstance(value, int | float) else 0.0
                    )
                    cursor += 1
            elif feature == "prompt_chars":
                values[row_index, cursor] = math.log1p(len(anchor.prompt))
                cursor += 1
            elif feature == "prompt_words":
                values[row_index, cursor] = math.log1p(len(anchor.prompt.split()))
                cursor += 1
            elif feature == "has_code":
                values[row_index, cursor] = float(
                    "```" in anchor.prompt or "def " in anchor.prompt or "function " in anchor.prompt
                )
                cursor += 1
            elif feature == "has_question":
                values[row_index, cursor] = float("?" in anchor.prompt)
                cursor += 1
            else:
                value = metrics.get(feature)
                if not isinstance(value, int | float) or not math.isfinite(float(value)):
                    value = stored_metrics.get(feature)
                values[row_index, cursor] = (
                    float(value) if isinstance(value, int | float) else _metric(anchor, feature)
                )
                cursor += 1
    return values, tuple(names)


def _outcome_rates(anchor: PublicAnchor, protocol_id: str | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for observation in anchor.observations:
        if (
            observation.outcome is not None
            and (protocol_id is None or observation.protocol_id == protocol_id)
        ):
            grouped[observation.model_id].append(float(bool(observation.outcome)))
    return {model: float(np.mean(values)) for model, values in grouped.items()}


def _models(anchors: Sequence[PublicAnchor], protocol_id: str | None = None) -> tuple[str, ...]:
    return tuple(sorted({model for anchor in anchors for model in _outcome_rates(anchor, protocol_id)}))


@dataclass(frozen=True)
class FittedAblation:
    strategy: str
    feature_names: tuple[str, ...]
    domain_names: tuple[str, ...]
    models: tuple[str, ...]
    coefficients: tuple[tuple[str, tuple[float, ...]], ...]

    def coefficient_map(self) -> dict[str, np.ndarray]:
        return {
            model: np.asarray(values, dtype=np.float64)
            for model, values in self.coefficients
        }


def fit_ablation(
    train: Sequence[PublicAnchor],
    *,
    strategy: str,
    protocol_id: str | None = None,
    ridge: float = 1.0,
    capability_profile: CapabilityProfile | None = None,
) -> FittedAblation:
    if strategy not in ABLATION_FEATURES:
        raise ValueError(f"unknown ablation strategy {strategy!r}")
    models = _models(train, protocol_id)
    if not models:
        raise ValueError("training anchors have no usable outcomes")
    domains = tuple(sorted({anchor.domain for anchor in train if anchor.domain}))
    coefficients: list[tuple[str, tuple[float, ...]]] = []
    for model in models:
        raw, names = _anchor_feature_values(
            train,
            domains=domains,
            features=ABLATION_FEATURES[strategy],
            target_model=model,
            protocol_id=protocol_id,
            capability_profile=capability_profile,
        )
        x = np.column_stack([np.ones(len(train)), raw])
        target: list[float] = []
        keep: list[int] = []
        for index, anchor in enumerate(train):
            rates = _outcome_rates(anchor, protocol_id)
            if model in rates:
                keep.append(index)
                target.append(rates[model])
        if not keep:
            continue
        coefficients.append(
            (
                model,
                tuple(float(value) for value in _logistic_fit(x[keep], np.asarray(target), ridge=ridge)),
            )
        )
    return FittedAblation(
        strategy=strategy,
        feature_names=names,
        domain_names=domains,
        models=tuple(model for model, _ in coefficients),
        coefficients=tuple(coefficients),
    )


def predict_ablation(
    fit: FittedAblation,
    anchors: Sequence[PublicAnchor],
    *,
    protocol_id: str | None = None,
    capability_profile: CapabilityProfile | None = None,
) -> list[dict[str, float]]:
    predictions: list[dict[str, float]] = []
    coefficients = fit.coefficient_map()
    per_model_rows: dict[str, np.ndarray] = {}
    for model in fit.models:
        raw, names = _anchor_feature_values(
            anchors,
            domains=fit.domain_names,
            features=ABLATION_FEATURES[fit.strategy],
            target_model=model,
            protocol_id=protocol_id,
            capability_profile=capability_profile,
        )
        if names != fit.feature_names:
            raise ValueError("ablation feature schema changed between fit and prediction")
        per_model_rows[model] = np.column_stack([np.ones(len(anchors)), raw])
    for row_index in range(len(anchors)):
        predictions.append(
            {
                model: float(_sigmoid(per_model_rows[model][row_index] @ coefficients[model]))
                for model in fit.models
            }
        )
    return predictions


def _bootstrap_ci(values: Sequence[float], *, seed: int = 17, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _choose_model(
    probabilities: Mapping[str, float],
    costs: Mapping[str, float],
    *,
    quality_threshold: float,
) -> str | None:
    eligible = [
        model
        for model, probability in probabilities.items()
        if probability >= quality_threshold and model in costs
    ]
    if eligible:
        return min(eligible, key=lambda model: (costs[model], -probabilities[model], model))
    available = [model for model in probabilities if model in costs]
    return max(available, key=lambda model: (probabilities[model], -costs[model], model)) if available else None


def evaluate_selection(
    train: Sequence[PublicAnchor],
    evaluation: Sequence[PublicAnchor],
    *,
    strategy: str,
        costs: Mapping[str, float],
        quality_threshold: float = 0.8,
        protocol_id: str | None = None,
        capability_profile: CapabilityProfile | None = None,
) -> dict[str, Any]:
    fit = fit_ablation(
        train,
        strategy=strategy,
        protocol_id=protocol_id,
        capability_profile=capability_profile,
    )
    predictions = predict_ablation(
        fit,
        evaluation,
        protocol_id=protocol_id,
        capability_profile=capability_profile,
    )
    selected_success: list[float] = []
    selected_cost: list[float] = []
    regret: list[float] = []
    selection_correct: list[float] = []
    unnecessary: list[float] = []
    missed: list[float] = []
    chosen_models: Counter[str] = Counter()
    evaluated = 0
    no_oracle = 0
    strongest = (
        capability_profile.order_weak_to_strong[-1]
        if capability_profile and capability_profile.order_weak_to_strong
        else None
    )
    for anchor, probabilities in zip(evaluation, predictions, strict=True):
        actual = _outcome_rates(anchor, protocol_id)
        if not actual:
            continue
        model = _choose_model(probabilities, costs, quality_threshold=quality_threshold)
        if model is None or model not in actual:
            continue
        evaluated += 1
        chosen_models[model] += 1
        selected_success.append(actual[model])
        selected_cost.append(float(costs[model]))
        successful = [candidate for candidate, value in actual.items() if value >= 0.5 and candidate in costs]
        if successful:
            oracle = min(successful, key=lambda candidate: (costs[candidate], candidate))
            oracle_cost = float(costs[oracle])
            regret.append(float(costs[model]) - oracle_cost)
            selection_correct.append(float(model == oracle))
            unnecessary.append(float(costs[model] > oracle_cost + 1e-12))
            missed.append(
                float(actual[model] < 0.5 and any(costs[candidate] > costs[model] for candidate in successful))
            )
        else:
            no_oracle += 1
    return {
        "strategy": strategy,
        "quality_threshold": quality_threshold,
        "count": evaluated,
        "selected_success_rate": float(np.mean(selected_success)) if selected_success else None,
        "selected_success_rate_95_ci": _bootstrap_ci(selected_success),
        "mean_cost_usd": float(np.mean(selected_cost)) if selected_cost else None,
        "mean_cost_95_ci": _bootstrap_ci(selected_cost, seed=19),
        "cost_per_success_usd": (
            float(np.mean(selected_cost) / np.mean(selected_success))
            if selected_cost and np.mean(selected_success) > 0
            else None
        ),
        "failure_rate": float(1.0 - np.mean(selected_success)) if selected_success else None,
        "oracle_cost_regret_usd": float(np.mean(regret)) if regret else None,
        "oracle_cost_regret_95_ci": _bootstrap_ci(regret, seed=23),
        "model_selection_accuracy": float(np.mean(selection_correct)) if selection_correct else None,
        "unnecessary_escalation_rate": float(np.mean(unnecessary)) if unnecessary else None,
        "missed_escalation_rate": float(np.mean(missed)) if missed else None,
        "oracle_covered_count": len(regret),
        "no_successful_model_count": no_oracle,
        "chosen_model_counts": dict(sorted(chosen_models.items())),
        "strongest_model_control": strongest,
    }


def _selection_vectors(
    train: Sequence[PublicAnchor],
    evaluation: Sequence[PublicAnchor],
    *,
    strategy: str,
    costs: Mapping[str, float],
    quality_threshold: float,
    protocol_id: str | None,
    capability_profile: CapabilityProfile | None,
) -> dict[str, dict[str, float]]:
    fit = fit_ablation(
        train,
        strategy=strategy,
        protocol_id=protocol_id,
        capability_profile=capability_profile,
    )
    predictions = predict_ablation(
        fit,
        evaluation,
        protocol_id=protocol_id,
        capability_profile=capability_profile,
    )
    vectors: dict[str, dict[str, float]] = {}
    for anchor, probabilities in zip(evaluation, predictions, strict=True):
        actual = _outcome_rates(anchor, protocol_id)
        model = _choose_model(probabilities, costs, quality_threshold=quality_threshold)
        if model is None or model not in actual:
            continue
        successful = [candidate for candidate, value in actual.items() if value >= 0.5 and candidate in costs]
        row = {
            "success": actual[model],
            "cost": float(costs[model]),
        }
        if successful:
            oracle = min(successful, key=lambda candidate: (costs[candidate], candidate))
            row.update(
                {
                    "regret": float(costs[model] - costs[oracle]),
                    "selection_accuracy": float(model == oracle),
                }
            )
        else:
            row.update({"regret": math.nan, "selection_accuracy": math.nan})
        vectors.setdefault(anchor.prompt_id, {}).update(row)
    return vectors


def compare_ablation_strategies(
    train: Sequence[PublicAnchor],
    evaluation: Sequence[PublicAnchor],
    *,
    costs: Mapping[str, float],
    quality_threshold: float = 0.8,
    protocol_id: str | None = None,
    capability_profile: CapabilityProfile | None = None,
) -> dict[str, Any]:
    """Return paired bootstrap deltas against scalar difficulty."""

    names = tuple(ABLATION_FEATURES)
    vectors = {
        name: _selection_vectors(
            train,
            evaluation,
            strategy=name,
            costs=costs,
            quality_threshold=quality_threshold,
            protocol_id=protocol_id,
            capability_profile=capability_profile,
        )
        for name in names
    }
    baseline = names[0]
    comparisons: dict[str, Any] = {}
    for name in names[1:]:
        common = sorted(set(vectors[baseline]) & set(vectors[name]))
        payload: dict[str, Any] = {"paired_count": len(common)}
        for metric in ("success", "cost", "regret", "selection_accuracy"):
            left = [vectors[name][prompt_id][metric] for prompt_id in common]
            right = [vectors[baseline][prompt_id][metric] for prompt_id in common]
            finite = [
                (a, b)
                for a, b in zip(left, right, strict=True)
                if math.isfinite(a) and math.isfinite(b)
            ]
            payload[f"{metric}_delta_95_ci"] = paired_bootstrap_difference(
                [a for a, _ in finite],
                [b for _, b in finite],
            )
        comparisons[f"{name}_vs_{baseline}"] = payload
    return {
        "baseline": baseline,
        "quality_threshold": quality_threshold,
        "comparisons": comparisons,
    }


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = 29,
    samples: int = 2000,
) -> list[float] | None:
    if len(left) != len(right) or not left:
        return None
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def equal_difficulty_specialization_slice(
    train: Sequence[PublicAnchor],
    evaluation: Sequence[PublicAnchor],
    *,
    costs: Mapping[str, float],
    capability_profile: CapabilityProfile | None = None,
    protocol_id: str | None = None,
    difficulty_center: float = 0.5,
    difficulty_tolerance: float = 0.08,
) -> dict[str, Any]:
    """Report low/high specialization at approximately equal difficulty."""

    def slice_metrics(anchor: PublicAnchor) -> dict[str, Any]:
        metrics = anchor.metric_map
        if capability_profile is not None and not isinstance(
            metrics.get("specialization_score"), int | float
        ):
            metrics = derive_anchor_metrics(
                anchor.observations,
                capability_profile=capability_profile,
                protocol_id=protocol_id,
            )
        return metrics

    candidates: list[PublicAnchor] = []
    metric_cache: dict[str, dict[str, Any]] = {}
    for anchor in evaluation:
        metrics = slice_metrics(anchor)
        metric_cache[anchor.prompt_id] = metrics
        difficulty = metrics.get("empirical_difficulty")
        specialization = metrics.get("specialization_score")
        if (
            isinstance(difficulty, int | float)
            and math.isfinite(float(difficulty))
            and isinstance(specialization, int | float)
            and math.isfinite(float(specialization))
            and abs(float(difficulty) - difficulty_center) <= difficulty_tolerance
        ):
            candidates.append(anchor)
    if not candidates:
        return {"count": 0, "difficulty_center": difficulty_center, "difficulty_tolerance": difficulty_tolerance}
    scores = sorted(float(metric_cache[anchor.prompt_id]["specialization_score"]) for anchor in candidates)
    midpoint = scores[len(scores) // 2]
    strongest = capability_profile.order_weak_to_strong[-1] if capability_profile else None
    groups: dict[str, list[PublicAnchor]] = {"low_specialization": [], "high_specialization": []}
    for anchor in candidates:
        key = (
            "high_specialization"
            if float(metric_cache[anchor.prompt_id]["specialization_score"]) >= midpoint
            else "low_specialization"
        )
        groups[key].append(anchor)
    result: dict[str, Any] = {
        "count": len(candidates),
        "difficulty_center": difficulty_center,
        "difficulty_tolerance": difficulty_tolerance,
        "specialization_median": midpoint,
        "groups": {},
    }
    for name, rows in groups.items():
        actual_success = [
            _outcome_rates(row, protocol_id)
            for row in rows
            if _outcome_rates(row, protocol_id)
        ]
        oracle_models: Counter[str] = Counter()
        strongest_success: list[float] = []
        for rates in actual_success:
            successful = [model for model, value in rates.items() if value >= 0.5 and model in costs]
            if successful:
                oracle_models[min(successful, key=lambda model: (costs[model], model))] += 1
            if strongest and strongest in rates:
                strongest_success.append(rates[strongest])
        result["groups"][name] = {
            "count": len(rows),
            "mean_difficulty": (
                float(np.mean([metric_cache[row.prompt_id]["empirical_difficulty"] for row in rows]))
                if rows
                else None
            ),
            "mean_specialization": (
                float(np.mean([metric_cache[row.prompt_id]["specialization_score"] for row in rows]))
                if rows
                else None
            ),
            "oracle_model_counts": dict(sorted(oracle_models.items())),
            "strongest_model_success_rate": float(np.mean(strongest_success)) if strongest_success else None,
        }
    return result


@dataclass(frozen=True)
class RaschAnchorFit:
    ability: tuple[tuple[str, float], ...]
    difficulty: tuple[tuple[str, float], ...]

    def probability(self, model_id: str, prompt_id: str) -> float:
        abilities = dict(self.ability)
        difficulties = dict(self.difficulty)
        return float(_sigmoid(abilities.get(model_id, 0.0) - difficulties.get(prompt_id, 0.0)))


def fit_rasch_anchors(
    anchors: Sequence[PublicAnchor],
    *,
    protocol_id: str | None = None,
    ridge: float = 1e-2,
    steps: int = 1200,
    learning_rate: float = 0.35,
) -> RaschAnchorFit:
    rows: list[tuple[str, str, float]] = []
    for anchor in anchors:
        for model, outcome in _outcome_rates(anchor, protocol_id).items():
            rows.append((model, anchor.prompt_id, outcome))
    if not rows:
        return RaschAnchorFit((), ())
    models = sorted({row[0] for row in rows})
    prompts = sorted({row[1] for row in rows})
    model_index = {model: index for index, model in enumerate(models)}
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}
    model_values = np.zeros(len(models), dtype=np.float64)
    prompt_values = np.zeros(len(prompts), dtype=np.float64)
    model_rows = np.asarray([model_index[row[0]] for row in rows], dtype=int)
    prompt_rows = np.asarray([prompt_index[row[1]] for row in rows], dtype=int)
    targets = np.asarray([row[2] for row in rows], dtype=np.float64)
    for _ in range(steps):
        logits = model_values[model_rows] - prompt_values[prompt_rows]
        probabilities = np.asarray(_sigmoid(logits), dtype=np.float64)
        residual = targets - probabilities
        gradient_model = np.bincount(model_rows, weights=residual, minlength=len(models)) - ridge * model_values
        gradient_prompt = -np.bincount(prompt_rows, weights=residual, minlength=len(prompts)) - ridge * prompt_values
        model_values += learning_rate * gradient_model / max(1.0, len(rows) / len(models))
        prompt_values += learning_rate * gradient_prompt / max(1.0, len(rows) / len(prompts))
        prompt_values -= prompt_values.mean()
    return RaschAnchorFit(
        ability=tuple((model, float(model_values[model_index[model]])) for model in models),
        difficulty=tuple((prompt, float(prompt_values[prompt_index[prompt]])) for prompt in prompts),
    )


def compare_empirical_vs_irt(
    anchors: Sequence[PublicAnchor],
    *,
    protocol_id: str | None = None,
    folds: int = 5,
    seed: int = 17,
) -> dict[str, Any]:
    """Compare prompt empirical priors and Rasch on held-out cells.

    Cells are held out, not entire prompts, so the comparison asks whether a
    latent prompt difficulty inferred from other observed model outcomes helps
    predict a held-out model outcome. This function does not silently claim an
    IRT win when the matrix is too sparse.
    """

    cells: list[tuple[str, str, float]] = []
    for anchor in anchors:
        for model, outcome in _outcome_rates(anchor, protocol_id).items():
            cells.append((anchor.prompt_id, model, outcome))
    if len(cells) < max(10, folds * 2):
        return {"status": "insufficient_outcomes", "cell_count": len(cells), "folds": folds}
    prompt_by_id = {anchor.prompt_id: anchor for anchor in anchors}
    order = sorted(
        range(len(cells)),
        key=lambda index: hashlib.sha256(f"{seed}:{cells[index][0]}:{cells[index][1]}".encode()).hexdigest(),
    )
    fold_indices = np.array_split(np.asarray(order, dtype=int), folds)
    empirical_losses: list[float] = []
    rasch_losses: list[float] = []
    evaluated = 0
    for fold in fold_indices:
        held_out = {int(index) for index in fold}
        train_cells = [row for index, row in enumerate(cells) if index not in held_out]
        train_by_prompt: dict[str, list[float]] = defaultdict(list)
        train_by_model: dict[str, list[float]] = defaultdict(list)
        for prompt_id, model, outcome in train_cells:
            train_by_prompt[prompt_id].append(outcome)
            train_by_model[model].append(outcome)
        train_anchors: list[PublicAnchor] = []
        for prompt_id, anchor in prompt_by_id.items():
            observations = tuple(
                AnchorObservation(
                    observation_id=f"cv:{prompt_id}:{model}",
                    model_id=model,
                    protocol_id=protocol_id or "cv",
                    outcome=outcome,
                )
                for prompt, model, outcome in train_cells
                if prompt == prompt_id
            )
            train_anchors.append(replace(anchor, observations=observations))
        fit = fit_rasch_anchors(train_anchors, protocol_id=protocol_id)
        for index in held_out:
            prompt_id, model, target = cells[index]
            prompt_prior = float(np.mean(train_by_prompt[prompt_id])) if train_by_prompt.get(prompt_id) else 0.5
            model_prior = float(np.mean(train_by_model[model])) if train_by_model.get(model) else 0.5
            empirical = min(1.0 - 1e-6, max(1e-6, 0.5 * (prompt_prior + model_prior)))
            rasch = fit.probability(model, prompt_id)
            empirical_losses.append(
                -(target * math.log(empirical) + (1.0 - target) * math.log(1.0 - empirical))
            )
            rasch_losses.append(
                -(target * math.log(rasch) + (1.0 - target) * math.log(1.0 - rasch))
            )
            evaluated += 1
    return {
        "status": "evaluated",
        "cell_count": len(cells),
        "evaluated_cells": evaluated,
        "folds": folds,
        "empirical_logloss": float(np.mean(empirical_losses)),
        "rasch_logloss": float(np.mean(rasch_losses)),
        "rasch_minus_empirical_logloss": float(np.mean(rasch_losses) - np.mean(empirical_losses)),
        "rasch_better": bool(np.mean(rasch_losses) < np.mean(empirical_losses)),
    }
