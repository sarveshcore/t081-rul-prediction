# ABOUTME: Tests for the shared PyTorch training loop and engine level cross validation.
# ABOUTME: Covers learning, early stopping, best weight restore, tuple inputs and seeding.
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rul.sequences import window_prepare
from rul.split import engine_split
from rul.train import cv_net, fit_net, predict_net, set_seed


class ToyLSTM(nn.Module):
    """Smallest useful sequence regressor: one LSTM layer and a linear head."""

    def __init__(self, n_features: int = 1, hidden: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class SumNet(nn.Module):
    """Two input net, for checking that tuple inputs reach forward as separate arguments."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(2))

    def forward(self, a, b):
        return self.scale[0] * a[:, -1, 0] + self.scale[1] * b[:, 0]


class ColumnNet(nn.Module):
    """A head that forgets to squeeze, so it returns shape (n, 1) instead of (n,)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)

    def forward(self, x):
        return self.head(x[:, -1])


class BatchNormNet(nn.Module):
    """A net with BatchNorm, which fails in train mode on a batch of one row."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(1)
        self.head = nn.Linear(1, 1)

    def forward(self, x):
        return self.head(self.norm(x[:, -1])).squeeze(-1)


def toy_data(n: int = 256, seed: int = 0):
    """Windows of noise whose target is three times the last value."""
    X = np.random.default_rng(seed).normal(size=(n, 10, 1)).astype(np.float32)
    return X, 3 * X[:, -1, 0]


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def test_fit_net_learns_toy_problem():
    X, y = toy_data()
    Xv, yv = toy_data(seed=1)
    set_seed(42)
    history = fit_net(ToyLSTM(), X, y, Xv, yv, epochs=60, lr=1e-2, batch_size=32)
    assert history["val_loss"][-1] < 0.2 * history["val_loss"][0]


def test_best_weights_restored():
    X, y = toy_data()
    Xv, yv = toy_data(seed=1)
    set_seed(42)
    net = ToyLSTM()
    history = fit_net(net, X, y, Xv, yv, epochs=20, lr=5e-2, batch_size=32)
    assert mse(predict_net(net, Xv), yv) == pytest.approx(min(history["val_loss"]), rel=1e-4)
    assert history["val_loss"][history["best_epoch"]] == min(history["val_loss"])


def test_early_stopping_stops():
    X, y = toy_data()
    set_seed(42)
    history = fit_net(ToyLSTM(), X, y, X, y, epochs=50, lr=0.0, patience=2)
    assert len(history["val_loss"]) == 3
    assert history["best_epoch"] == 0


def test_tuple_inputs():
    X, _ = toy_data()
    b = np.ones((len(X), 3), dtype=np.float32)
    y = X[:, -1, 0] + 2 * b[:, 0]
    set_seed(42)
    net = SumNet()
    fit_net(net, (X, b), y, (X, b), y, epochs=200, lr=5e-2, batch_size=64)
    prediction = predict_net(net, (X, b))
    assert prediction.shape == (len(X),)
    assert mse(prediction, y) < 1e-2


def test_same_seed_same_predictions():
    X, y = toy_data()
    predictions = []
    for _ in range(2):
        set_seed(42)
        net = ToyLSTM()
        fit_net(net, X, y, X, y, epochs=5, batch_size=32)
        predictions.append(predict_net(net, X))
    np.testing.assert_array_equal(*predictions)


def test_fit_net_rejects_column_output():
    X, y = toy_data()
    with pytest.raises(ValueError, match="one value per row"):
        fit_net(ColumnNet(), X, y, X, y, epochs=1)


def test_predict_net_rejects_column_output():
    X, _ = toy_data()
    with pytest.raises(ValueError, match="one value per row"):
        predict_net(ColumnNet(), X)


def test_one_row_last_batch_is_skipped():
    X, y = toy_data(n=65)
    set_seed(42)
    history = fit_net(BatchNormNet(), X, y, X, y, epochs=2, batch_size=64)
    assert len(history["train_loss"]) == 2


def engines_frame() -> pd.DataFrame:
    """Ten engines of 20 to 29 cycles, two noisy sensors linear in RUL, index shuffled."""
    rng = np.random.default_rng(42)
    rows = [(u, c) for u in range(1, 11) for c in range(1, 20 + u)]
    df = pd.DataFrame(rows, columns=["unit", "cycle"])
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["s_a"] = df["rul"] + rng.normal(size=len(df))
    df["s_b"] = -0.5 * df["rul"] + rng.normal(size=len(df))
    return df.sample(frac=1, random_state=42)


def spy_prepare(calls: list):
    """A window prepare function that records which units it fits on and applies to."""
    inner = window_prepare(["s_a", "s_b"], window=5)

    def prepare(fit_df, apply_df):
        calls.append((set(fit_df["unit"]), set(apply_df["unit"])))
        return inner(fit_df, apply_df)
    return prepare


def run_cv(calls: list, n_splits: int = 5) -> pd.DataFrame:
    return cv_net(lambda: ToyLSTM(n_features=2), engines_frame(), spy_prepare(calls),
                  n_splits=n_splits, epochs=3, batch_size=64)


def test_cv_net_returns_one_row_per_fold():
    folds = run_cv([])
    assert list(folds["fold"]) == [0, 1, 2, 3, 4]
    assert {"rmse", "mae", "nasa_score", "rmse_late", "best_epoch", "n_rows"} <= set(folds)
    assert np.isfinite(folds[["rmse", "mae"]].to_numpy()).all()
    assert folds["n_rows"].sum() == len(engines_frame())


def test_cv_net_scores_every_unit_once(monkeypatch):
    calls = []

    def recording_split(df, **kwargs):
        inner, es = engine_split(df, **kwargs)
        calls.append(("early stopping", set(es["unit"])))
        return inner, es
    monkeypatch.setattr("rul.train.engine_split", recording_split)
    run_cv(calls)
    scored, es = [], None
    for fit, applied in calls:
        if fit == "early stopping":
            es = applied
        elif applied not in (fit, es):
            scored.append(applied)
    assert sorted(u for units in scored for u in units) == list(range(1, 11))


def test_scored_and_early_stopping_engines_never_fit():
    calls = []
    run_cv(calls)
    for fold in range(5):
        (fit, train), (fit_es, es), (fit_val, val) = calls[3 * fold: 3 * fold + 3]
        assert fit == train == fit_es == fit_val
        assert es and not es & fit
        assert val and not val & fit and not val & es
