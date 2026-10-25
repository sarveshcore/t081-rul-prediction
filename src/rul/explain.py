# ABOUTME: Explanation helpers for fitted RUL regressors.
# ABOUTME: Maps feature columns to their source sensor and permutes whole sensor groups.
import re

import numpy as np
import pandas as pd


def sensor_group(column: str) -> str:
    """Sensor a feature column is derived from (s_2_mean_5 -> s_2); other columns map to themselves."""
    match = re.match(r"s_\d+", column)
    return match.group() if match else column


def group_permutation_importance(
    model, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 42
) -> pd.Series:
    """Mean RMSE increase when every column of one sensor group is shuffled with the same order."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    def rmse(data):
        return float(np.sqrt(np.mean((model.predict(data) - y) ** 2)))

    base = rmse(X)
    increase = {}
    for group, columns in X.columns.groupby(X.columns.map(sensor_group)).items():
        drops = []
        for _ in range(n_repeats):
            # One shared row order keeps the group's columns consistent with each other.
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]
            drops.append(rmse(shuffled) - base)
        increase[group] = np.mean(drops)
    return pd.Series(increase, name="rmse_increase").sort_values(ascending=False)
