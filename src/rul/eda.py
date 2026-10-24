# ABOUTME: Descriptive sensor statistics for the FD001 training-data EDA.
# ABOUTME: Records the evidence for dropping constant and nearly constant binary sensors.
import pandas as pd


def sensor_summary(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Summarize variation and apply the documented training-only sensor-selection rule."""
    values = df[sensors]
    summary = pd.DataFrame({
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "max": values.max(),
        "n_unique": values.nunique(),
        "dominant_fraction": [values[s].value_counts(normalize=True).iloc[0] for s in sensors],
    })
    summary.index.name = "sensor"
    summary["relative_std"] = summary["std"] / summary["mean"].abs().replace(0, float("nan"))
    summary["decision"] = "keep"
    summary.loc[summary["n_unique"] == 1, "decision"] = "drop: constant"
    binary = (summary["n_unique"] == 2) & (summary["dominant_fraction"] >= 0.98)
    summary.loc[binary, "decision"] = "drop: near-constant binary"
    return summary
