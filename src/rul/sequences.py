# ABOUTME: Fixed length sensor windows for sequence models on C-MAPSS trajectories.
# ABOUTME: One window per cycle, built inside each engine and left padded with its first row.
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler


def _scaled(df: pd.DataFrame, sensors: list[str], scaler) -> tuple[pd.DataFrame, StandardScaler]:
    """Sort by unit and cycle, then scale sensors, fitting the scaler only when none is given."""
    df = df.sort_values(["unit", "cycle"])
    if scaler is None:
        scaler = StandardScaler().fit(df[sensors])
    values = scaler.transform(df[sensors]).astype(np.float32)
    return df.assign(**dict(zip(sensors, values.T))), scaler


def _engine_windows(values: np.ndarray, window: int) -> np.ndarray:
    """All windows of one engine, one ending at each cycle, padded with the first row."""
    padded = np.vstack([np.repeat(values[:1], window - 1, axis=0), values])
    return sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)


def make_windows(df: pd.DataFrame, sensors: list[str], window: int = 30, scaler=None,
                 target: str = "rul") -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Return X (n, window, n_sensors), y, unit and the scaler; fits the scaler if None."""
    df, scaler = _scaled(df, sensors, scaler)
    X = np.concatenate([_engine_windows(g[sensors].to_numpy(), window)
                        for _, g in df.groupby("unit", sort=True)])
    y = df[target].to_numpy(dtype=np.float32)
    return X.astype(np.float32), y, df["unit"].to_numpy(), scaler


def last_windows(df: pd.DataFrame, sensors: list[str], window: int = 30,
                 scaler=None) -> tuple[np.ndarray, pd.Index]:
    """One window per unit ending at its last cycle, sorted by unit to match test RUL."""
    df, _ = _scaled(df, sensors, scaler)
    groups = df.groupby("unit", sort=True)
    X = np.stack([_engine_windows(g[sensors].to_numpy(), window)[-1] for _, g in groups])
    return X.astype(np.float32), pd.Index(list(groups.groups), name="unit")


def window_prepare(sensors: list[str], window: int = 30) -> Callable:
    """Build a cv_net prepare function: fit the scaler on fit_df, window apply_df."""
    def prepare(fit_df: pd.DataFrame, apply_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler().fit(fit_df[sensors])
        X, y, _, _ = make_windows(apply_df, sensors, window, scaler=scaler)
        return X, y
    return prepare
