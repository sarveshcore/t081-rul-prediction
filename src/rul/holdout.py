# ABOUTME: Helpers for scoring models once on the held out FD001 test engines.
# ABOUTME: Each engine's last row, and a paired bootstrap over engines averaged across seeds.
import numpy as np
import pandas as pd


def last_row_positions(df: pd.DataFrame) -> np.ndarray:
    """Positions of each unit's last cycle in df sorted by unit and cycle, in unit order."""
    units = df.sort_values(["unit", "cycle"])["unit"].to_numpy()
    return np.flatnonzero(np.append(units[1:] != units[:-1], True))


def seed_bootstrap(preds_a: np.ndarray, preds_b: np.ndarray, truth: np.ndarray,
                   n_boot: int = 10_000, seed: int = 42) -> dict:
    """Mean RMSE over the seeds of model a minus that of model b, with a 95% interval.

    preds_a and preds_b have shape (seeds, engines), or (engines,) for a single fit. Each draw
    resamples the engines once and scores every seed of both models on that same resample, so
    the interval matches the reported mean over seeds rather than the score of an averaged
    prediction.
    """
    truth = np.asarray(truth, dtype=float)
    err_a, err_b = (np.atleast_2d(np.asarray(p, dtype=float)) - truth for p in (preds_a, preds_b))

    def mean_rmse(error, idx):
        # error is (seeds, engines) and idx is (draws, engines), so this is (seeds, draws).
        return np.sqrt((error[:, idx] ** 2).mean(axis=-1)).mean(axis=0)

    draws = np.random.default_rng(seed).integers(0, len(truth), size=(n_boot, len(truth)))
    whole = np.arange(len(truth))[None, :]
    diff = (mean_rmse(err_a, whole) - mean_rmse(err_b, whole))[0]
    diffs = mean_rmse(err_a, draws) - mean_rmse(err_b, draws)
    return {"rmse_difference": float(diff), "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5))}
