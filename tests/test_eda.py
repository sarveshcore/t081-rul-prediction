# ABOUTME: Tests for descriptive FD001 sensor statistics.
# ABOUTME: Distinguishes exact constants and dominant binary signals from small-scale variation.
import pandas as pd

from rul.eda import sensor_summary


def test_sensor_summary_distinguishes_constant_quantized_and_small_scale_signals():
    """Raw standard deviation alone must not discard sensors with small physical units."""
    df = pd.DataFrame({
        "flat": [5.0] * 100,
        "binary": [21.61] * 98 + [21.60] * 2,
        "varied": [i / 1_000_000 for i in range(100)],
        "balanced": [0, 1] * 50,
    })
    summary = sensor_summary(df, list(df.columns))
    assert "decision" not in summary.columns
    assert summary.index.tolist() == list(df.columns)
    assert summary.loc["varied", "n_unique"] == 100
    assert summary.loc["balanced", "dominant_fraction"] == 0.5
    assert summary.loc["binary", "min"] == 21.60
    assert summary.loc["binary", "max"] == 21.61
    assert summary.loc["flat", "n_unique"] == 1
    assert summary.loc["binary", "dominant_fraction"] == 0.98
