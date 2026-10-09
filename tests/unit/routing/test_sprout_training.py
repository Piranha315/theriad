from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from theriad.routing.sprout_training import (
    MODEL_COLUMNS,
    SproutExample,
    adapt_projected_row,
    assert_disjoint_splits,
    balanced_sample_indices,
    deduplicate_examples,
    model_score_metrics,
    source_family,
)


def _row(prompt: str = "Solve this"):
    row = {"key": prompt, "dataset": "TIGER-Lab/MMLU-Pro", "prompt": prompt}
    row.update({model: index / 12 for index, model in enumerate(MODEL_COLUMNS)})
    return row


def test_source_family_collapses_nested_datasets():
    assert source_family("rungalileo/ragbench/hotpotqa") == "ragbench"
    assert source_family("TAUR-Lab/MuSR/team_allocation") == "musr"
    assert source_family("lighteval/MATH/all/test") == "math"


def test_group_holdout_and_official_test_policy():
    row = _row()
    row["dataset"] = "TAUR-Lab/MuSR"
    held_out = adapt_projected_row(row, official_split="train")
    official_test = adapt_projected_row(row, official_split="test")
    assert held_out is not None and held_out.split == "validation"
    assert official_test is not None and official_test.split == "test"


def test_deduplication_keeps_test_prompt_out_of_training():
    train = adapt_projected_row(_row("same"), official_split="train")
    test = adapt_projected_row(_row("same"), official_split="test")
    assert train is not None and test is not None
    output, dropped = deduplicate_examples([train, test])
    assert len(output) == 1 and output[0].split == "test"
    assert dropped == {"train_duplicate": 1}
    assert_disjoint_splits(output)


def test_balanced_sampling_is_uniform_across_imbalanced_families():
    examples = [
        SproutExample(str(i), (0.0, 1.0), "a", "a", "train", "train", str(i))
        for i in range(100)
    ]
    examples.append(SproutExample("b", (0.0, 1.0), "b", "b", "train", "train", "b"))
    indices = balanced_sample_indices(examples, count=4000, seed=7)
    counts = Counter(examples[index].source_family for index in indices)
    assert abs(counts["a"] - counts["b"]) < 200


def test_metrics_measure_model_selection_regret():
    target = np.asarray([[0.2, 0.9, 0.5], [0.8, 0.1, 0.4]])
    prediction = np.asarray([[0.1, 0.8, 0.3], [0.2, 0.9, 0.3]])
    mask = np.ones_like(target, dtype=bool)
    metrics = model_score_metrics(prediction, target, mask)
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["top1_regret"] == pytest.approx(0.35)
    assert metrics["selected_mean_score"] == 0.5
