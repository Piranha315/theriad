from __future__ import annotations

import numpy as np
import pytest

from scripts.compare_gist_difficulty import (
    make_folds,
    oof_affine_calibration,
    oof_mean,
    regression_metrics,
)


def test_oof_affine_calibration_recovers_linear_mapping():
    raw = np.linspace(0.0, 1.0, 20)
    target = 4.0 * raw - 2.0
    prediction, parameters = oof_affine_calibration(raw, target, make_folds(20))

    assert all(row["slope"] == pytest.approx(4.0) for row in parameters)
    assert all(row["intercept"] == pytest.approx(-2.0) for row in parameters)
    assert np.max(np.abs(prediction - target)) <= 4.0 / 19.0 + 1e-12
    assert prediction.min() >= target.min()
    assert prediction.max() <= target.max()


def test_oof_mean_never_uses_held_out_target():
    target = np.asarray([0.0, 0.0, 0.0, 10.0])
    folds = [np.asarray([0, 1]), np.asarray([2, 3])]

    assert oof_mean(target, folds) == pytest.approx([5.0, 5.0, 0.0, 0.0])


def test_regression_metrics_average_tied_ranks():
    metric = regression_metrics(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    assert metric["rmse"] == 0.0
    assert metric["spearman"] == pytest.approx(1.0)
