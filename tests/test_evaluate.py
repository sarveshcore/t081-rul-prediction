# ABOUTME: Tests for RUL metrics in rul.evaluate, checked against hand worked values.
# ABOUTME: Covers RMSE, MAE, the NASA asymmetric score and the early vs late stage split.
import math

import numpy as np
import pandas as pd
import pytest

from rul.evaluate import evaluate, nasa_score


def test_nasa_early_and_late():
    # d = -13 and d = +10 both give exp(1) - 1
    assert nasa_score([100, 100], [87, 110]) == pytest.approx(2 * (math.e - 1))


def test_nasa_penalises_late_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_nasa_perfect_is_zero():
    assert nasa_score([10, 20, 30], [10, 20, 30]) == 0.0


def test_rmse_mae():
    m = evaluate([10, 20, 30, 40], [12, 18, 33, 40])
    assert m["rmse"] == pytest.approx(math.sqrt(17 / 4))
    assert m["mae"] == pytest.approx(7 / 4)


def test_stage_split():
    # true 10 and 40 are late (<= 50), true 60 and 100 are early
    m = evaluate([10, 40, 60, 100], [20, 40, 60, 90])
    assert m["rmse_late"] == pytest.approx(math.sqrt(50))
    assert m["mae_late"] == pytest.approx(5)
    assert m["rmse_early"] == pytest.approx(math.sqrt(50))
    assert m["mae_early"] == pytest.approx(5)


def test_rul_at_late_cut_counts_as_late():
    m = evaluate([50, 100], [60, 100])
    assert m["mae_late"] == pytest.approx(10)
    assert m["mae_early"] == pytest.approx(0)


def test_empty_stage_is_nan():
    m = evaluate([60, 100], [70, 90])
    assert math.isnan(m["rmse_late"])
    assert math.isnan(m["mae_late"])


def test_returns_every_metric():
    m = evaluate([10, 60], [12, 55])
    assert set(m) == {
        "rmse", "mae", "nasa_score", "rmse_early", "mae_early", "rmse_late", "mae_late",
    }


def test_accepts_series_with_index():
    true = pd.Series([10, 40, 60, 100], index=[5, 9, 2, 7])
    pred = pd.Series([20, 40, 60, 90], index=[1, 2, 3, 4])
    from_series = evaluate(true, pred)
    from_arrays = evaluate(np.array([10, 40, 60, 100]), np.array([20, 40, 60, 90]))
    assert from_series == pytest.approx(from_arrays)
