# ABOUTME: Hybrid remaining useful life models combining sequence encoders and engineered features.
# ABOUTME: Two-branch neural networks and residual boosting on C-MAPSS trajectories.
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from rul.evaluate import evaluate
from rul.features import FeatureScaler, feature_groups
from rul.models import build_model
from rul.sequences import make_windows
from rul.split import engine_folds, engine_split
from rul.train import SEED, fit_net, predict_net, set_seed


def engineered_columns(sensors: list[str], windows: tuple[int, ...] = (5, 10, 20)) -> list[str]:
    """List of 142 engineered degradation columns matching the Phase 1 feature set."""
    groups = feature_groups(sensors, windows)
    return ["cycle", *sensors, *groups["rolling"], *groups["slope"], *groups["health"]]


class HybridNet(nn.Module):
    """Two-branch network: encodes a sensor window and concatenates engineered features to an MLP head."""

    def __init__(
        self,
        encoder: nn.Module,
        n_features: int,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Drop unused sequence encoder head so dead parameters are not kept in optimizer
        if hasattr(encoder, "head"):
            encoder.head = nn.Identity()
        self.encoder = encoder
        self.n_features = n_features
        self.out_dim = encoder.out_dim
        in_dim = encoder.out_dim + n_features
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, windows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass taking (windows, features) tensors, returning shape (batch,)."""
        z_seq = self.encoder.encode(windows)
        combined = torch.cat([z_seq, features], dim=-1)
        return self.head(self.drop(combined)).squeeze(-1)


def hybrid_prepare(
    sensors: list[str],
    window: int = 30,
    windows_tuple: tuple[int, ...] = (5, 10, 20),
    feature_cols: list[str] | None = None,
    target: str = "rul",
) -> Callable:
    """Build a cv_net prepare function returning ((windows, features), y).

    Fits the sequence StandardScaler, the FeatureScaler, and a feature normalizer
    on fit_df only, guaranteeing no information from apply_df reaches the transformations.
    """
    if feature_cols is None:
        cols = engineered_columns(sensors, windows_tuple)
    else:
        cols = list(feature_cols)

    def prepare(fit_df: pd.DataFrame, apply_df: pd.DataFrame) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        # Sequence scaler and FeatureScaler fit on fit_df only
        seq_scaler = StandardScaler().fit(fit_df[sensors])
        feat_scaler = FeatureScaler.fit(fit_df, sensors, windows_tuple)

        # Feature normalization fits on fit_df only
        fit_feat = feat_scaler.transform(fit_df.sort_values(["unit", "cycle"]))
        feat_norm = StandardScaler().fit(fit_feat[cols])

        # Windows and features computed on apply_df
        apply_sorted = apply_df.sort_values(["unit", "cycle"])
        X_seq, y, _, _ = make_windows(apply_sorted, sensors, window=window, scaler=seq_scaler, target=target)

        apply_feat = feat_scaler.transform(apply_sorted)
        X_feat = feat_norm.transform(apply_feat[cols]).astype(np.float32)

        return (X_seq, X_feat), y

    return prepare


def cv_residual_hybrid(
    build_seq_net: Callable[[], nn.Module],
    df: pd.DataFrame,
    sensors: list[str],
    window: int = 30,
    windows_tuple: tuple[int, ...] = (5, 10, 20),
    xgb_params: dict | None = None,
    n_splits: int = 5,
    seed: int = SEED,
    es_frac: float = 0.2,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    res_scale: float = 25.0,
) -> pd.DataFrame:
    """Cross-validation for the residual hybrid: sequence model predicting XGBoost residuals.

    XGBoost fits on engineered features on inner fold training engines. The sequence model
    predicts the residual (y - xgb_pred) from 30-cycle sensor windows. Predictions are summed
    and clipped to [0, 125].
    """
    if xgb_params is None:
        xgb_params = {
            "subsample": 1.0,
            "reg_lambda": 5,
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
        }

    cols = engineered_columns(sensors, windows_tuple)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(engine_folds(df, n_splits=n_splits)):
        inner_df, es_df = engine_split(df.iloc[train_idx], val_frac=es_frac, seed=seed)
        val_df = df.iloc[val_idx]

        # Ensure consistent engine and cycle ordering across feature engineering and windows
        inner_df = inner_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        es_df = es_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        val_df = val_df.sort_values(["unit", "cycle"]).reset_index(drop=True)

        # 1. Feature engineering and scaling fit strictly on inner_df
        feat_scaler = FeatureScaler.fit(inner_df, sensors, windows_tuple)
        inner_feat = feat_scaler.transform(inner_df)
        es_feat = feat_scaler.transform(es_df)
        val_feat = feat_scaler.transform(val_df)

        # 2. Fit XGBoost on inner fold
        xgb = build_model("xgboost", **xgb_params)
        xgb.fit(inner_feat[cols], inner_feat["rul"])

        pred_inner_xgb = xgb.predict(inner_feat[cols])
        pred_es_xgb = xgb.predict(es_feat[cols])
        pred_val_xgb = xgb.predict(val_feat[cols])

        res_inner = (inner_feat["rul"].to_numpy(dtype=np.float32) - pred_inner_xgb).astype(np.float32)
        res_es = (es_feat["rul"].to_numpy(dtype=np.float32) - pred_es_xgb).astype(np.float32)

        # 3. Sequence windows
        seq_scaler = StandardScaler().fit(inner_df[sensors])
        X_seq_inner, _, _, _ = make_windows(inner_df, sensors, window=window, scaler=seq_scaler)
        X_seq_es, _, _, _ = make_windows(es_df, sensors, window=window, scaler=seq_scaler)
        X_seq_val, y_val, _, _ = make_windows(val_df, sensors, window=window, scaler=seq_scaler)

        # 4. Train sequence model to predict residuals
        set_seed(seed)
        net = build_seq_net()
        history = fit_net(
            net,
            X_seq_inner,
            res_inner,
            X_seq_es,
            res_es,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            seed=seed,
            y_scale=res_scale,
        )

        res_val_pred = predict_net(net, X_seq_val, y_scale=res_scale)
        final_val_pred = np.clip(pred_val_xgb + res_val_pred, 0.0, 125.0)

        metrics = evaluate(y_val, final_val_pred)
        rows.append({
            "fold": fold,
            **metrics,
            "best_epoch": history["best_epoch"],
            "n_rows": len(y_val),
        })

    return pd.DataFrame(rows)
