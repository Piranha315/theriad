from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_sprout_costs import (
    MODEL_COLUMNS,
    choices_with_tolerance,
    policy_metrics,
    uniform_metrics,
)


def test_tolerance_selects_cheapest_model_near_predicted_best():
    prediction = np.asarray([[0.90, 0.87, 0.40], [0.40, 0.80, 0.79]])
    expected_cost = np.asarray([10.0, 2.0, 1.0])
    assert choices_with_tolerance(prediction, expected_cost, 0.05).tolist() == [1, 2]
    assert choices_with_tolerance(prediction, expected_cost, 0.0).tolist() == [0, 1]


def test_policy_metrics_uses_per_prompt_selected_score_and_cost():
    score = np.asarray([[0.2, 0.9], [0.8, 0.1]])
    cost = np.asarray([[0.001, 0.004], [0.002, 0.005]])
    result = policy_metrics("test", np.asarray([1, 0]), score, cost)
    assert result["mean_score"] == pytest.approx(0.85)
    assert result["cost_per_1000_requests_usd"] == pytest.approx(3.0)
    assert result["successful_equivalent_tasks_per_dollar"] == pytest.approx(0.85 / 0.003)
    assert result["model_selection_counts"] == {
        MODEL_COLUMNS[0]: 1,
        MODEL_COLUMNS[1]: 1,
    }


def test_uniform_metrics_averages_all_prompt_model_cells():
    score = np.asarray([[0.2, 0.9], [0.8, 0.1]])
    cost = np.asarray([[0.001, 0.004], [0.002, 0.005]])
    result = uniform_metrics(score, cost)
    assert result["mean_score"] == pytest.approx(0.5)
    assert result["cost_per_1000_requests_usd"] == pytest.approx(3.0)
