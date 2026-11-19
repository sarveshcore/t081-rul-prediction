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

    # Each row is plotted as its difference from the baseline of its own run; rows scored against
    # different labels have no reference and are left out.
    df = df[df["reference"].notna() & (df["ablation"] != df["reference"])].reset_index(drop=True)
    df_plot = df.iloc[::-1].reset_index(drop=True)

    colors = []
    for name in df_plot["ablation"]:
        if "Window" in name:
            colors.append("#ff7f0e")
        elif "Target" in name:
            colors.append("#d62728")
        elif "Regularisation" in name:
            colors.append("#9467bd")
        elif "Inputs" in name:
            colors.append("#2ca02c")
        elif "Hybrid" in name:
            colors.append("#8c564b")
        else:
            colors.append("#7f7f7f")

    _fig, ax = plt.subplots(figsize=(10, max(5.0, len(df_plot) * 0.45)), dpi=300)
    y_positions = np.arange(len(df_plot))
    ax.axvline(0, color="#1f77b4", linestyle="--", linewidth=1.4, label="Baseline of the same run")
    ax.barh(y_positions, df_plot["rmse_delta"], xerr=df_plot["rmse_std"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.6, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
            height=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot["ablation"])
    ax.set_xlabel("CV RMSE minus baseline RMSE (cycles; error bar = the ablation's fold std)", fontweight="bold")
    ax.set_title("Single-factor ablations against the baseline of the same run", fontweight="bold", pad=12)
    # Labels go past the end of each error bar so they never sit on it.
    for y, d, sd in zip(y_positions, df_plot["rmse_delta"], df_plot["rmse_std"]):
        x = d + sd + 0.12 if d >= 0 else d - sd - 0.12
        ax.text(x, y, f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=8)
    lo = (df_plot["rmse_delta"] - df_plot["rmse_std"]).min()
    hi = (df_plot["rmse_delta"] + df_plot["rmse_std"]).max()
    ax.set_xlim(min(lo, 0) - 0.8, hi + 0.8)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_file = FIGURES_DIR / "ablation_cv_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {Path(out_file).relative_to(ROOT)}")


if __name__ == "__main__":
    plot_ablations_comparison()
