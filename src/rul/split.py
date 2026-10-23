# ABOUTME: Engine level train and validation splits for C-MAPSS data.
# ABOUTME: Every split groups by `unit`, so no engine's cycles land on both sides.
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def engine_split(
    df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a random fraction of whole engines as validation."""
    units = np.random.default_rng(seed).permutation(np.sort(df["unit"].unique()))
    n_val = max(1, round(val_frac * len(units)))
    is_val = df["unit"].isin(units[:n_val])
    return df[~is_val], df[is_val]


def engine_folds(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return positional (train_idx, val_idx) pairs for GroupKFold grouped by unit, reusable."""
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df["unit"]))


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unit, taken at that unit's final cycle."""
    last = df.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    return last.reset_index(drop=True)
