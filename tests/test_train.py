# ABOUTME: Tests for the shared PyTorch training loop and engine level cross validation.
# ABOUTME: Covers learning, early stopping, best weight restore, tuple inputs and seeding.
import numpy as np
import pytest
import torch
from torch import nn

from rul.train import fit_net, predict_net, set_seed


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
