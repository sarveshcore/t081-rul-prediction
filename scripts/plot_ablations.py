# ABOUTME: Generates figures/ablation_cv_comparison.png for Issue #37.
# ABOUTME: Plots 5-fold CV RMSE with fold standard deviation error bars against baseline.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

ABLATIONS_CSV = RESULTS_DIR / "ablations.csv"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_ablations_comparison():
    if not ABLATIONS_CSV.exists():
        print(f"File {ABLATIONS_CSV} does not exist yet.")
        return

    df = pd.read_csv(ABLATIONS_CSV)
    if len(df) == 0:
        return

    # Invert order so baseline appears at the top
    df_plot = df.iloc[::-1].reset_index(drop=True)

    base_row = df[df["ablation"].str.contains("Baseline")].iloc[0]
    base_rmse = base_row["rmse_mean"]
    base_std = base_row["rmse_std"]

    # Category colors
    colors = []
    for name in df_plot["ablation"]:
        if "Baseline" in name:
            colors.append("#1f77b4")  # blue
        elif "Window" in name:
            colors.append("#ff7f0e")  # orange
        elif "Target" in name:
            colors.append("#d62728")  # red
        elif "Regularisation" in name or "Reg" in name:
            colors.append("#9467bd")  # purple
        elif "Inputs" in name or "SHAP" in name:
            colors.append("#2ca02c")  # green
        elif "Hybrid" in name:
            colors.append("#8c564b")  # brown
        else:
            colors.append("#7f7f7f")

    _fig, ax = plt.subplots(figsize=(10, max(5.0, len(df_plot) * 0.45)), dpi=300)

    y_positions = np.arange(len(df_plot))

    # Reference baseline line and +-1 std shaded band
    ax.axvline(base_rmse, color="#1f77b4", linestyle="--", linewidth=1.4, alpha=0.9, label=f"Baseline Transformer ({base_rmse:.2f} cycles)")
    ax.axvspan(base_rmse - base_std, base_rmse + base_std, color="#1f77b4", alpha=0.12, label="Baseline $\\pm 1\\sigma$ fold spread")

    # Plot horizontal error bars
    bars = ax.barh(
        y_positions,
        df_plot["rmse_mean"],
        xerr=df_plot["rmse_std"],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        height=0.6,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot["ablation"], fontweight="medium")
    ax.set_xlabel("Five-Fold Engine Cross-Validation RMSE (cycles, lower is better)", fontweight="bold")
    ax.set_title("Single-Factor Ablation Studies: Validation RMSE ($\\pm 1\\sigma$ Fold Spread)", fontweight="bold", pad=12)

    # Annotate values and delta from baseline
    for idx, (bar, row) in enumerate(zip(bars, df_plot.iterrows())):
        r = row[1]
        val = r["rmse_mean"]
        std = r["rmse_std"]
        diff = val - base_rmse
        diff_str = f" (+{diff:.2f})" if diff > 0.05 else (f" ({diff:.2f})" if diff < -0.05 else " (ref)")
        text = rf"{val:.2f} $\pm$ {std:.2f}{diff_str}"
        ax.text(val + std + 0.3, idx, text, va="center", ha="left", fontsize=8.5, color="#222222")

    min_x = max(10, df_plot["rmse_mean"].min() - 2)
    max_x = df_plot["rmse_mean"].max() + df_plot["rmse_std"].max() + 3.5
    ax.set_xlim(min_x, max_x)

    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_file = FIGURES_DIR / "ablation_cv_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_ablations_comparison()
