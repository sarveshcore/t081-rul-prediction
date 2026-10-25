# ABOUTME: Tests for sensor grouping and grouped permutation importance in rul.explain.
# ABOUTME: A model that uses one sensor must lose accuracy only when that sensor is shuffled.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from rul.explain import group_permutation_importance, sensor_group


def test_sensor_group_maps_derived_columns_to_their_sensor():
    columns = ["cycle", "health_index", "s_2", "s_2_mean_5", "s_21_std_10", "s_21_slope_20"]
    assert [sensor_group(c) for c in columns] == [
        "cycle", "health_index", "s_2", "s_2", "s_21", "s_21"]


def test_only_the_used_sensor_group_matters():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["s_2", "s_2_mean_5", "s_3"])
    y = X["s_2"] + X["s_2_mean_5"]
    model = LinearRegression().fit(X, y)

    importance = group_permutation_importance(model, X, y, n_repeats=3)

    assert list(importance.index) == ["s_2", "s_3"]
    assert importance["s_2"] > 1
    assert abs(importance["s_3"]) < 1e-6
