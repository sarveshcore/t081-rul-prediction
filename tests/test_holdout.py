# ABOUTME: Tests for scoring models on the held out FD001 test engines.
# ABOUTME: Covers picking each engine's last row and the paired bootstrap averaged over seeds.
import numpy as np
import pandas as pd
import pytest

from rul.holdout import last_row_positions, seed_bootstrap


def test_last_row_positions_index_the_unit_cycle_sorted_order():
    # Shuffled rows: unit 1 has cycles 1 to 3, unit 2 has cycles 1 to 2.
    df = pd.DataFrame({"unit": [2, 1, 1, 2, 1], "cycle": [2, 3, 1, 1, 2]})
    np.testing.assert_array_equal(last_row_positions(df), [2, 4])


def test_seed_bootstrap_is_zero_for_identical_models():
    truth = np.arange(20, dtype=float)
    pred = truth + np.linspace(-3, 3, 20)
    out = seed_bootstrap(np.stack([pred, pred]), pred, truth, n_boot=200)
    assert out == pytest.approx({"rmse_difference": 0.0, "low": 0.0, "high": 0.0}, abs=1e-12)


def test_seed_bootstrap_averages_rmse_over_seeds_not_predictions():
    truth = np.zeros(10)
    # Seed errors +2 and -4: mean RMSE is 3, but the averaged prediction would score 1.
    preds_a = np.stack([truth + 2, truth - 4])
    out = seed_bootstrap(preds_a, truth, truth, n_boot=200)
    assert out["rmse_difference"] == 3.0
    assert out["low"] == out["high"] == 3.0


def test_seed_bootstrap_averages_rmse_over_the_seeds_of_both_models():
    truth = np.zeros(10)
    preds_b = np.stack([truth + 2, truth - 4])
    out = seed_bootstrap(truth[None, :], preds_b, truth, n_boot=200)
    assert out["rmse_difference"] == -3.0
    assert out["low"] == out["high"] == -3.0


def test_seed_bootstrap_interval_contains_the_point_difference():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 125, 100)
    preds_a = truth + rng.normal(0, 10, (3, 100))
    pred_b = truth + rng.normal(0, 14, 100)
    out = seed_bootstrap(preds_a, pred_b, truth)
    assert out["low"] < out["rmse_difference"] < out["high"] < 0
