"""Pure data and metric utilities for SPROUT model-performance training.

The live Theriad router must not import PyTorch, Sentence Transformers, or the
Hugging Face data stack.  This module keeps the dataset contract, split policy,
balanced sampler, and evaluation metrics dependency-light and unit-testable.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SPROUT_REPO = "CARROT-LLM-Routing/SPROUT"
SPROUT_REVISION = "d54e3cda2189ca1ed13a877a44ed34736c4b6920"
SPROUT_LICENSE = None

MODEL_COLUMNS = (
    "aws-claude-3-5-sonnet-v1",
    "aws-titan-text-premier-v1",
    "openai-gpt-4o",
    "openai-gpt-4o-mini",
    "wxai-granite-3-2b-instruct-8k-max-tokens",
    "wxai-granite-3-8b-instruct-8k-max-tokens",
    "wxai-llama-3-1-70b-instruct",
    "wxai-llama-3-1-8b-instruct",
    "wxai-llama-3-2-1b-instruct",
    "wxai-llama-3-2-3b-instruct",
    "wxai-llama-3-3-70b-instruct",
    "wxai-llama-3-405b-instruct",
    "wxai-mixtral-8x7b-instruct-v01",
)

DEFAULT_VALIDATION_FAMILIES = ("musr", "ragbench")
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class SproutExample:
    """One prompt with a masked vector of per-model judge scores."""

    prompt: str
    scores: tuple[float | None, ...]
    source: str
    source_family: str
    split: str
    official_split: str
    record_id: str

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["scores"] = list(self.scores)
        return row


def normalise_prompt(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(normalise_prompt(prompt).encode("utf-8")).hexdigest()


def source_family(source: str) -> str:
    """Collapse SPROUT's dataset paths into six non-overlapping families."""

    lowered = source.lower()
    if lowered.startswith("rungalileo/ragbench/"):
        return "ragbench"
    if lowered.startswith("taur-lab/musr"):
        return "musr"
    if lowered.startswith("lighteval/math"):
        return "math"
    if lowered.startswith("idavidrein/gpqa"):
        return "gpqa"
    if lowered.startswith("tiger-lab/mmlu-pro"):
        return "mmlu_pro"
    if lowered.startswith("openhermes/teknium"):
        return "openhermes"
    return lowered or "unknown"


def _score(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def adapt_projected_row(
    row: dict[str, Any],
    *,
    official_split: str,
    validation_families: Iterable[str] = DEFAULT_VALIDATION_FAMILIES,
) -> SproutExample | None:
    """Adapt a compact Parquet projection rather than full model responses."""

    prompt = normalise_prompt(row.get("prompt"))
    source = str(row.get("dataset") or "")
    family = source_family(source)
    scores = tuple(_score(row.get(model)) for model in MODEL_COLUMNS)
    if not prompt or sum(score is not None for score in scores) < 2:
        return None
    if official_split == "test":
        split = "test"
    elif family in set(validation_families):
        split = "validation"
    else:
        split = "train"
    record_id = str(row.get("key") or prompt_key(prompt)[:16])
    return SproutExample(
        prompt=prompt,
        scores=scores,
        source=source,
        source_family=family,
        split=split,
        official_split=official_split,
        record_id=record_id,
    )


def deduplicate_examples(
    examples: Iterable[SproutExample],
) -> tuple[list[SproutExample], dict[str, int]]:
    """Remove exact prompt overlap with test > grouped validation > train."""

    grouped: dict[str, list[SproutExample]] = defaultdict(list)
    for example in examples:
        grouped[prompt_key(example.prompt)].append(example)
    output: list[SproutExample] = []
    dropped = Counter()
    for rows in grouped.values():
        winner = max(rows, key=lambda row: SPLIT_PRIORITY[row.split])
        output.append(winner)
        for row in rows:
            if row is not winner:
                dropped[f"{row.split}_duplicate"] += 1
    output.sort(key=lambda row: (SPLIT_PRIORITY[row.split], row.record_id))
    return output, dict(dropped)


def assert_disjoint_splits(examples: Iterable[SproutExample]) -> None:
    seen: dict[str, str] = {}
    for example in examples:
        key = prompt_key(example.prompt)
        previous = seen.setdefault(key, example.split)
        if previous != example.split:
            raise ValueError(f"prompt overlap between {previous} and {example.split}")


def balanced_sample_indices(
    examples: list[SproutExample], *, count: int, seed: int
) -> list[int]:
    """Sample source families uniformly, then prompts uniformly within family."""

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        buckets[example.source_family].append(index)
    if not buckets:
        return []
    families = sorted(buckets)
    rng = random.Random(seed)
    return [rng.choice(buckets[rng.choice(families)]) for _ in range(count)]


def _rankdata(values: np.ndarray) -> np.ndarray:
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


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 2:
        return float("nan")
    prediction_rank = _rankdata(prediction)
    target_rank = _rankdata(target)
    prediction_rank -= prediction_rank.mean()
    target_rank -= target_rank.mean()
    denominator = math.sqrt(
        float(np.sum(prediction_rank**2) * np.sum(target_rank**2))
    )
    return float(np.sum(prediction_rank * target_rank) / denominator) if denominator else 0.0


def model_score_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Measure score fit and, more importantly, per-prompt model selection."""

    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    valid_error = prediction[mask] - target[mask]
    if not valid_error.size:
        raise ValueError("at least one score must be observed")

    spearman: list[float] = []
    regrets: list[float] = []
    selected_scores: list[float] = []
    oracle_scores: list[float] = []
    random_scores: list[float] = []
    top1_matches = 0
    evaluated_prompts = 0
    for predicted_row, target_row, mask_row in zip(prediction, target, mask, strict=True):
        valid = np.flatnonzero(mask_row)
        if not valid.size:
            continue
        local_prediction = predicted_row[valid]
        local_target = target_row[valid]
        selected = int(np.argmax(local_prediction))
        selected_score = float(local_target[selected])
        oracle = float(np.max(local_target))
        selected_scores.append(selected_score)
        oracle_scores.append(oracle)
        random_scores.append(float(np.mean(local_target)))
        regrets.append(oracle - selected_score)
        top1_matches += int(selected_score == oracle)
        evaluated_prompts += 1
        rank = _spearman(local_prediction, local_target)
        if math.isfinite(rank):
            spearman.append(rank)

    return {
        "prompt_count": evaluated_prompts,
        "observed_score_count": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(valid_error**2))),
        "mae": float(np.mean(np.abs(valid_error))),
        "mean_prompt_spearman": float(np.mean(spearman)) if spearman else 0.0,
        "top1_accuracy": top1_matches / evaluated_prompts,
        "top1_regret": float(np.mean(regrets)),
        "selected_mean_score": float(np.mean(selected_scores)),
        "oracle_mean_score": float(np.mean(oracle_scores)),
        "random_mean_score": float(np.mean(random_scores)),
    }
