# ABOUTME: Tests for the descriptive FD001 sensor-selection rule.
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
    assert summary.loc["flat", "decision"] == "drop: constant"
    assert summary.loc["binary", "decision"] == "drop: near-constant binary"
    assert summary.loc["varied", "decision"] == "keep"
    assert summary.loc["balanced", "decision"] == "keep"
    assert summary.loc["flat", "n_unique"] == 1
    assert summary.loc["binary", "dominant_fraction"] == 0.98
