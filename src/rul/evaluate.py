# ABOUTME: Metrics for remaining useful life predictions on C-MAPSS.
# ABOUTME: RMSE, MAE, the NASA asymmetric score, and early vs late stage error.
import numpy as np


def nasa_score(y_true, y_pred) -> float:
    """Sum of exp(-d/13) - 1 for early and exp(d/10) - 1 for late predictions, d = pred - true."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


def _rmse_mae(error: np.ndarray) -> tuple[float, float]:
    """Return (rmse, mae) of an error array, or (nan, nan) when it is empty."""
    if error.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))


def evaluate(y_true, y_pred, late_cut: int = 50) -> dict:
    """Return rmse, mae, nasa_score and rmse/mae for early (> late_cut) and late stages."""
    y_true = np.ravel(np.asarray(y_true, dtype=float))
    y_pred = np.ravel(np.asarray(y_pred, dtype=float))
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has {y_true.size} values but y_pred has {y_pred.size}")
    error = y_pred - y_true
    late = y_true <= late_cut
    rmse, mae = _rmse_mae(error)
    rmse_early, mae_early = _rmse_mae(error[~late])
    rmse_late, mae_late = _rmse_mae(error[late])
    return {
        "rmse": rmse,
        "mae": mae,
        "nasa_score": nasa_score(y_true, y_pred),
        "rmse_early": rmse_early,
        "mae_early": mae_early,
        "rmse_late": rmse_late,
        "mae_late": mae_late,
    }
