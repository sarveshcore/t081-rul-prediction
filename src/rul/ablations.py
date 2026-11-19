# ABOUTME: Dataset coverage diagnostics and ablation models for NASA C-MAPSS FD001.
# ABOUTME: Covers window-length coverage, padding trade-offs, RUL clipping, and single-factor ablations.
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from rul.data import SETTINGS, USEFUL_SENSORS, add_rul, load_fd, load_train_fd
from rul.features import FeatureScaler
from rul.hybrid import engineered_columns
from rul.nets import build_net


def dataset_coverage(window_lengths: tuple[int, ...] = (15, 30, 50)) -> pd.DataFrame:
    """Calculate window coverage and padding counts on train cycles and test engines.

    For each window length w:
    - Train cycles kept without padding: cycles with index >= w inside each engine.
    - Train cycles padded: (w - 1) * number of train engines (100).
    - Test engines kept without padding: engines with total observed lifetime >= w.
    - Test engines padded: engines with total observed lifetime < w (requiring left padding at last cycle).
    """
    train, test, _ = load_fd("FD001")
    n_train_engines = train["unit"].nunique()
    total_train_cycles = len(train)
    test_lens = test.groupby("unit")["cycle"].max()
    n_test_engines = len(test_lens)

    rows = []
    for w in window_lengths:
        train_kept = sum(max(0, int(l) - w + 1) for l in train.groupby("unit")["cycle"].max())
        train_padded = n_train_engines * (w - 1)
        test_kept = int((test_lens >= w).sum())
        test_padded = int((test_lens < w).sum())
        rows.append({
            "window": w,
            "train_cycles_total": total_train_cycles,
            "train_cycles_kept": train_kept,
            "train_kept_pct": 100.0 * train_kept / total_train_cycles,
            "train_windows_padded": train_padded,
            "test_engines_total": n_test_engines,
            "test_engines_kept": test_kept,
            "test_engines_padded": test_padded,
            "shortest_test_engine": int(test_lens.min()),
        })
    return pd.DataFrame(rows)


def cap_distribution(cap: int = 125) -> dict[str, float]:
    """Summary statistics of training RUL before and after clipping at cap."""
    raw_df = add_rul(load_train_fd("FD001"), cap=None)
    capped_df = add_rul(load_train_fd("FD001"), cap=cap)

    raw_rul = raw_df["rul"]
    capped_rul = capped_df["rul"]

    return {
        "raw_mean": float(raw_rul.mean()),
        "raw_std": float(raw_rul.std()),
        "raw_median": float(raw_rul.median()),
        "raw_max": float(raw_rul.max()),
        "capped_mean": float(capped_rul.mean()),
        "capped_std": float(capped_rul.std()),
        "capped_median": float(capped_rul.median()),
        "capped_max": float(capped_rul.max()),
        "n_capped": int((raw_rul > cap).sum()),
        "pct_capped": float(100.0 * (raw_rul > cap).mean()),
    }


def regime_variation() -> pd.DataFrame:
    """Show setting statistics confirming FD001 operates in a single condition."""
    train = load_train_fd("FD001")
    rows = []
    for s in SETTINGS:
        series = train[s]
        rows.append({
            "setting": s,
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "n_unique": int(series.nunique()),
        })
    return pd.DataFrame(rows)


class SequenceMLPNet(nn.Module):
    """Sequence branch alone through the Two-Branch MLP head (no engineered features)."""

    def __init__(
        self,
        n_features: int = len(USEFUL_SENSORS),
        d_model: int = 64,
        layers: int = 1,
        heads: int = 4,
        mlp_hidden: int = 64,
        encoder_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = build_net(
            "transformer",
            n_features=n_features,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=encoder_dropout,
        )
        self.encoder.head = nn.Identity()
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(windows)
        return self.head(self.drop(z)).squeeze(-1)


class FeatureMLPNet(nn.Module):
    """Feature branch alone through the Two-Branch MLP head (no sequence encoder)."""

    def __init__(
        self,
        n_features: int = 142,
        mlp_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(n_features, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(features)).squeeze(-1)


def feature_prepare(
    sensors: list[str] = USEFUL_SENSORS,
    windows_tuple: tuple[int, ...] = (5, 10, 20),
    target: str = "rul",
) -> Callable:
    """Build a cv_net prepare function returning features standardized on fit_df only."""
    cols = engineered_columns(sensors, windows_tuple)

    def prepare(fit_df: pd.DataFrame, apply_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feat_scaler = FeatureScaler.fit(fit_df, sensors, windows_tuple)
        fit_feat = feat_scaler.transform(fit_df.sort_values(["unit", "cycle"]))
        feat_norm = StandardScaler().fit(fit_feat[cols])

        apply_sorted = apply_df.sort_values(["unit", "cycle"])
        apply_feat = feat_scaler.transform(apply_sorted)
        X = feat_norm.transform(apply_feat[cols]).astype(np.float32)
        y = apply_sorted[target].to_numpy(dtype=np.float32)
        return X, y

    return prepare
