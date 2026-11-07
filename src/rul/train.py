# ABOUTME: One seeded PyTorch training loop shared by every Phase 2 sequence model.
# ABOUTME: Adam on MSE with early stopping, prediction, and engine level cross validation.
import copy
import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from rul.evaluate import evaluate
from rul.split import engine_folds, engine_split

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so the same seed gives the same weights and batches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _tensors(inputs) -> tuple[torch.Tensor, ...]:
    """Turn one array or a tuple of arrays into a tuple of float tensors."""
    arrays = inputs if isinstance(inputs, tuple) else (inputs,)
    return tuple(torch.as_tensor(np.asarray(a), dtype=torch.float32) for a in arrays)


def _loss(net: nn.Module, inputs: tuple[torch.Tensor, ...], y: torch.Tensor) -> float:
    """Mean squared error of the net on all of inputs, without tracking gradients."""
    net.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(net(*inputs), y))


def fit_net(net: nn.Module, train_inputs, y_train, val_inputs, y_val, *, epochs: int = 100,
            batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 0.0,
            patience: int = 10, grad_clip: float | None = 1.0, seed: int = SEED) -> dict:
    """Train with Adam on MSE, stop after patience epochs without improvement, keep best weights."""
    X, Xv = _tensors(train_inputs), _tensors(val_inputs)
    y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
    yv = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_state, best_loss = copy.deepcopy(net.state_dict()), float("inf")

    for epoch in range(epochs):
        net.train()
        batch_losses = []
        for idx in torch.randperm(len(y), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(*(x[idx] for x in X)), y[idx])
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())
        val_loss = _loss(net, Xv, yv)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss, history["best_epoch"] = val_loss, epoch
            best_state = copy.deepcopy(net.state_dict())
        elif epoch - history["best_epoch"] >= patience:
            break

    net.load_state_dict(best_state)
    return history


def predict_net(net: nn.Module, inputs, batch_size: int = 1024) -> np.ndarray:
    """Predict in evaluation mode, in batches, returning a flat float64 array."""
    X = _tensors(inputs)
    net.eval()
    with torch.no_grad():
        parts = [net(*(x[i:i + batch_size] for x in X)) for i in range(0, len(X[0]), batch_size)]
    return torch.cat(parts).double().numpy()


def cv_net(build_net: Callable[[], nn.Module], df: pd.DataFrame, prepare: Callable, *,
           n_splits: int = 5, seed: int = SEED, es_frac: float = 0.2, **fit_kwargs) -> pd.DataFrame:
    """GroupKFold by unit; early stopping uses engines held out of each fold's training engines.

    prepare(fit_df, apply_df) -> (inputs, y) fits any scaler on fit_df only. Every row of each
    held out fold is scored, so pass df with the target already capped as in Phase 1 CV.
    """
    rows = []
    for fold, (train_idx, val_idx) in enumerate(engine_folds(df, n_splits=n_splits)):
        inner, es = engine_split(df.iloc[train_idx], val_frac=es_frac, seed=seed)
        train_inputs, y_train = prepare(inner, inner)
        es_inputs, y_es = prepare(inner, es)
        val_inputs, y_val = prepare(inner, df.iloc[val_idx])
        set_seed(seed)
        net = build_net()
        history = fit_net(net, train_inputs, y_train, es_inputs, y_es, seed=seed, **fit_kwargs)
        metrics = evaluate(y_val, predict_net(net, val_inputs))
        rows.append({"fold": fold, **metrics, "best_epoch": history["best_epoch"],
                     "n_rows": len(y_val)})
    return pd.DataFrame(rows)
