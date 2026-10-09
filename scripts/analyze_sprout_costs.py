#!/usr/bin/env python3
"""Evaluate the trained SPROUT quality router on a cost/quality frontier.

Pricing uses Table 2 of the CARROT paper (dollars per million input/output
tokens), not present-day replacement-model prices.  Selection costs are
estimated from the prepared training split only.  Cost-aware thresholds are
chosen on the grouped validation split and then evaluated once on the official
test split.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sprout_router import PARQUET_SHARDS, parquet_alias  # noqa: E402
from theriad.routing.sprout_training import MODEL_COLUMNS  # noqa: E402

DEFAULT_ARTIFACT = ROOT / "artifacts" / "gist-sprout-router"

# CARROT paper Table 2 and the CARROT Hugging Face organization page.
# Values are dollars per million input/output tokens.
SPROUT_PRICES = {
    "aws-claude-3-5-sonnet-v1": (3.0, 15.0),
    "aws-titan-text-premier-v1": (0.5, 1.5),
    "openai-gpt-4o": (2.5, 10.0),
    "openai-gpt-4o-mini": (0.15, 0.6),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.1, 0.1),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.2, 0.2),
    "wxai-llama-3-1-70b-instruct": (0.9, 0.9),
    "wxai-llama-3-1-8b-instruct": (0.2, 0.2),
    "wxai-llama-3-2-1b-instruct": (0.06, 0.06),
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-llama-3-3-70b-instruct": (0.9, 0.9),
    "wxai-llama-3-405b-instruct": (3.5, 3.5),
    "wxai-mixtral-8x7b-instruct-v01": (0.6, 0.6),
}

# The repository's current constant conflicts only for Titan.  This alternative
# is reported as a sensitivity check rather than silently choosing one value.
REPOSITORY_TITAN_PRICE = (0.8, 3.2)


def projection_sql(output: Path) -> str:
    fields = ["key"]
    for index, model in enumerate(MODEL_COLUMNS):
        fields.extend(
            [
                f'"{model}".num_input_tokens AS "m{index}_input"',
                f'"{model}".num_output_tokens AS "m{index}_output"',
            ]
        )
    escaped = str(output).replace("'", "''")
    return f"COPY (SELECT {', '.join(fields)} FROM data) TO '{escaped}' (FORMAT JSON)"


def project_token_counts(split: str, shard: str, output: Path) -> None:
    if shutil.which("npx") is None:
        raise SystemExit("npx is required for the Parquet projection step")
    command = [
        "npx",
        "-y",
        "-p",
        "parquetlens",
        "-p",
        "@parquetlens/sql",
        "parquetlens",
        parquet_alias(split, shard),
        "--sql",
        projection_sql(output),
    ]
    subprocess.run(command, check=True)


def parse_count(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("missing token count")
    count = int(value)
    if count < 0:
        raise ValueError("negative token count")
    return count


def load_token_counts() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    counts: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    with tempfile.TemporaryDirectory(prefix="theriad-sprout-cost-") as temporary:
        directory = Path(temporary)
        for split, shards in PARQUET_SHARDS.items():
            for shard in shards:
                path = directory / f"{split}-{shard}.jsonl"
                project_token_counts(split, shard, path)
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        inputs = np.asarray(
                            [parse_count(row[f"m{index}_input"]) for index in range(len(MODEL_COLUMNS))],
                            dtype=np.float64,
                        )
                        outputs = np.asarray(
                            [parse_count(row[f"m{index}_output"]) for index in range(len(MODEL_COLUMNS))],
                            dtype=np.float64,
                        )
                        key = (split, str(row["key"]))
                        if key in counts:
                            raise ValueError(f"duplicate token-count key: {key}")
                        counts[key] = (inputs, outputs)
    return counts


def load_prepared(artifact: Path, split: str) -> list[dict[str, Any]]:
    path = artifact / "prepared" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cost_matrix(
    rows: list[dict[str, Any]],
    token_counts: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    prices: dict[str, tuple[float, float]],
) -> np.ndarray:
    input_prices = np.asarray([prices[model][0] for model in MODEL_COLUMNS])
    output_prices = np.asarray([prices[model][1] for model in MODEL_COLUMNS])
    result = []
    for row in rows:
        key = (str(row["official_split"]), str(row["record_id"]))
        try:
            inputs, outputs = token_counts[key]
        except KeyError as exc:
            raise KeyError(f"no token counts for prepared row {key}") from exc
        result.append((inputs * input_prices + outputs * output_prices) / 1_000_000.0)
    return np.asarray(result)


def targets(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray([[np.nan if value is None else value for value in row["scores"]] for row in rows])
    if np.isnan(matrix).any():
        raise ValueError("cost analysis currently requires all 13 scores")
    return matrix


def infer(artifact: Path, prompts: list[str], *, device: str, batch_size: int) -> np.ndarray:
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(artifact / "encoder"), device=device)
    embedding = encoder.encode(
        prompts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    state = load_file(artifact / "performance_head.safetensors")
    raw = embedding @ state["weight"].numpy().T + state["bias"].numpy()
    return 1.0 / (1.0 + np.exp(-raw))


def choices_with_tolerance(
    prediction: np.ndarray,
    expected_model_cost: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    threshold = prediction.max(axis=1, keepdims=True) - tolerance
    eligible = prediction >= threshold
    candidate_cost = np.where(eligible, expected_model_cost[None, :], np.inf)
    return np.argmin(candidate_cost, axis=1)


def policy_metrics(
    name: str,
    choices: np.ndarray,
    score: np.ndarray,
    cost: np.ndarray,
) -> dict[str, Any]:
    index = np.arange(len(choices))
    selected_score = score[index, choices]
    selected_cost = cost[index, choices]
    mean_score = float(np.mean(selected_score))
    mean_cost = float(np.mean(selected_cost))
    return {
        "name": name,
        "prompt_count": len(choices),
        "mean_score": mean_score,
        "cost_per_request_usd": mean_cost,
        "cost_per_1000_requests_usd": 1000.0 * mean_cost,
        "successful_equivalent_tasks_per_dollar": mean_score / mean_cost,
        "cost_per_successful_equivalent_task_usd": mean_cost / mean_score,
        "model_selection_counts": {
            MODEL_COLUMNS[model]: count
            for model, count in sorted(Counter(choices.tolist()).items())
        },
    }


def uniform_metrics(score: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    mean_score = float(np.mean(score))
    mean_cost = float(np.mean(cost))
    return {
        "name": "Uniform model selection",
        "prompt_count": len(score),
        "mean_score": mean_score,
        "cost_per_request_usd": mean_cost,
        "cost_per_1000_requests_usd": 1000.0 * mean_cost,
        "successful_equivalent_tasks_per_dollar": mean_score / mean_cost,
        "cost_per_successful_equivalent_task_usd": mean_cost / mean_score,
        "model_selection_counts": {model: len(score) / len(MODEL_COLUMNS) for model in MODEL_COLUMNS},
    }


def select_tolerance(
    prediction: np.ndarray,
    score: np.ndarray,
    cost: np.ndarray,
    expected_model_cost: np.ndarray,
    quality_tolerance: float,
) -> tuple[float, dict[str, Any]]:
    quality_choices = np.argmax(prediction, axis=1)
    quality_score = policy_metrics("quality", quality_choices, score, cost)["mean_score"]
    candidates = []
    for tolerance in np.linspace(0.0, 1.0, 201):
        choices = choices_with_tolerance(prediction, expected_model_cost, float(tolerance))
        metrics = policy_metrics("candidate", choices, score, cost)
        if metrics["mean_score"] >= quality_score - quality_tolerance:
            candidates.append((metrics["cost_per_request_usd"], float(tolerance), metrics))
    if not candidates:
        raise RuntimeError("no tolerance satisfies the validation quality guardrail")
    _, tolerance, metrics = min(candidates, key=lambda row: row[0])
    return tolerance, metrics


def fixed_model_metrics(
    model_index: int, score: np.ndarray, cost: np.ndarray, *, label: str | None = None
) -> dict[str, Any]:
    choices = np.full(len(score), model_index, dtype=np.int64)
    return policy_metrics(label or f"Always {MODEL_COLUMNS[model_index]}", choices, score, cost)


def add_comparisons(policy: dict[str, Any], comparators: dict[str, dict[str, Any]]) -> None:
    policy["comparisons"] = {}
    for name, comparator in comparators.items():
        comparator_cost = comparator["cost_per_request_usd"]
        policy["comparisons"][name] = {
            "cost_savings_fraction": 1.0 - policy["cost_per_request_usd"] / comparator_cost,
            "score_difference": policy["mean_score"] - comparator["mean_score"],
        }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact_dir
    prepared = {split: load_prepared(artifact, split) for split in ("train", "validation", "test")}
    token_counts = load_token_counts()
    cost = {
        split: cost_matrix(rows, token_counts, prices=SPROUT_PRICES)
        for split, rows in prepared.items()
    }
    score = {split: targets(rows) for split, rows in prepared.items() if split != "train"}
    prediction = {
        split: infer(
            artifact,
            [str(row["prompt"]) for row in prepared[split]],
            device=args.device,
            batch_size=args.batch_size,
        )
        for split in ("validation", "test")
    }
    expected_model_cost = cost["train"].mean(axis=0)

    validation_quality_choices = np.argmax(prediction["validation"], axis=1)
    test_quality_choices = np.argmax(prediction["test"], axis=1)
    validation_quality = policy_metrics(
        "Quality-first embedding router",
        validation_quality_choices,
        score["validation"],
        cost["validation"],
    )
    test_quality = policy_metrics(
        "Quality-first embedding router",
        test_quality_choices,
        score["test"],
        cost["test"],
    )

    guarded_policies = {}
    for tolerance_points in args.quality_tolerances:
        tolerance_score = tolerance_points / 100.0
        selected_delta, validation_metrics = select_tolerance(
            prediction["validation"],
            score["validation"],
            cost["validation"],
            expected_model_cost,
            tolerance_score,
        )
        test_choices = choices_with_tolerance(
            prediction["test"], expected_model_cost, selected_delta
        )
        key = f"within_{tolerance_points:g}_points"
        test_metrics = policy_metrics(
            f"Cost-aware router (validation guardrail: −{tolerance_points:g} point)",
            test_choices,
            score["test"],
            cost["test"],
        )
        guarded_policies[key] = {
            "selected_prediction_tolerance": selected_delta,
            "validation": validation_metrics,
            "test": test_metrics,
        }

    validation_fixed_scores = score["validation"].mean(axis=0)
    best_fixed_index = int(np.argmax(validation_fixed_scores))
    cheapest_fixed_index = int(np.argmin(expected_model_cost))
    fixed = {
        "best_fixed_selected_on_validation": fixed_model_metrics(
            best_fixed_index,
            score["test"],
            cost["test"],
            label=f"Best fixed model (selected on validation): {MODEL_COLUMNS[best_fixed_index]}",
        ),
        "cheapest_fixed_selected_on_train_cost": fixed_model_metrics(
            cheapest_fixed_index,
            score["test"],
            cost["test"],
            label=f"Cheapest fixed model: {MODEL_COLUMNS[cheapest_fixed_index]}",
        ),
        "gpt_4o": fixed_model_metrics(
            MODEL_COLUMNS.index("openai-gpt-4o"), score["test"], cost["test"]
        ),
        "gpt_4o_mini": fixed_model_metrics(
            MODEL_COLUMNS.index("openai-gpt-4o-mini"), score["test"], cost["test"]
        ),
        "claude_3_5_sonnet": fixed_model_metrics(
            MODEL_COLUMNS.index("aws-claude-3-5-sonnet-v1"), score["test"], cost["test"]
        ),
    }
    uniform = uniform_metrics(score["test"], cost["test"])
    comparators = {
        "uniform_model_selection": uniform,
        "gpt_4o": fixed["gpt_4o"],
        "best_fixed_model": fixed["best_fixed_selected_on_validation"],
        "quality_first_router": test_quality,
    }
    add_comparisons(test_quality, {key: value for key, value in comparators.items() if key != "quality_first_router"})
    for policy in guarded_policies.values():
        add_comparisons(policy["test"], comparators)

    # Sensitivity to the conflicting Titan constant.
    sensitivity_prices = dict(SPROUT_PRICES)
    sensitivity_prices["aws-titan-text-premier-v1"] = REPOSITORY_TITAN_PRICE
    sensitivity_cost_test = cost_matrix(prepared["test"], token_counts, prices=sensitivity_prices)
    titan_sensitivity = {
        "quality_first_cost_per_1000_requests_usd": policy_metrics(
            "quality", test_quality_choices, score["test"], sensitivity_cost_test
        )["cost_per_1000_requests_usd"],
        "uniform_cost_per_1000_requests_usd": uniform_metrics(
            score["test"], sensitivity_cost_test
        )["cost_per_1000_requests_usd"],
    }

    return {
        "schema_version": 1,
        "pricing_basis": {
            "units": "USD per million input/output tokens",
            "primary": "CARROT paper Table 2 and CARROT Hugging Face organization table",
            "prices": {model: list(SPROUT_PRICES[model]) for model in MODEL_COLUMNS},
            "price_vintage": "benchmark contemporaneous; not current replacement-model pricing",
            "known_conflict": {
                "model": "aws-titan-text-premier-v1",
                "paper_and_hf": list(SPROUT_PRICES["aws-titan-text-premier-v1"]),
                "repository_current_constant": list(REPOSITORY_TITAN_PRICE),
                "sensitivity": titan_sensitivity,
            },
        },
        "method": {
            "cost_formula": "(input_tokens * input_price + output_tokens * output_price) / 1,000,000",
            "selection_cost_estimate": "mean per-model realized cost on prepared training prompts only",
            "threshold_selection": "chosen on grouped validation, evaluated once on official test",
            "quality_tolerances": [f"{value:g} percentage point(s)" for value in args.quality_tolerances],
            "router_overhead_excluded": True,
        },
        "model_expected_cost_from_train": {
            model: float(expected_model_cost[index]) for index, model in enumerate(MODEL_COLUMNS)
        },
        "validation_quality_first": validation_quality,
        "test": {
            "uniform": uniform,
            "quality_first_router": test_quality,
            "cost_aware_routers": guarded_policies,
            "fixed_models": fixed,
        },
        "limitations": [
            "Prices reproduce the SPROUT benchmark and are not today's deployable catalog prices.",
            "The quality head was trained without a cost objective; cost-aware policies are post-hoc.",
            "Expected model cost uses a training-set mean rather than a prompt-conditioned output-token predictor.",
            "Router compute, network, retry, and observability overhead are excluded.",
            "SPROUT has no declared dataset license and is not a defense-domain operational benchmark.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quality-tolerances", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if any(value < 0 for value in args.quality_tolerances):
        raise SystemExit("quality tolerances cannot be negative")
    output = args.output or args.artifact_dir / "cost_analysis.json"
    result = analyze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
