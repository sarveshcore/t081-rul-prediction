# ABOUTME: Generates figures/ablation_coverage_padding.png for Issue #37 and the report.
# ABOUTME: Plots window length coverage/padding trade-offs and RUL distributions before/after cap.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rul.ablations import cap_distribution, dataset_coverage
from rul.data import add_rul, load_train_fd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

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


def plot_dataset_diagnostics():
    cov_df = dataset_coverage()
    cap_stats = cap_distribution(125)

    raw_df = add_rul(load_train_fd("FD001"), cap=None)
    capped_df = add_rul(load_train_fd("FD001"), cap=125)

    _fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # 1. Left Plot: Window Length Coverage & Padding Trade-off
    ax1 = axes[0]
    windows = cov_df["window"].tolist()
    x = np.arange(len(windows))
    width = 0.35

    kept = cov_df["train_cycles_kept"].to_numpy()
    padded = cov_df["train_windows_padded"].to_numpy()

    b1 = ax1.bar(x - width / 2, kept, width, label="Kept without pad", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + width / 2, padded, width, label="Padded windows", color="#d95f02", edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Number of Training Cycles / Windows", fontweight="bold")
    ax1.set_xlabel("Sequence Window Length ($w$)", fontweight="bold")
    ax1.set_title("(a) Window Length vs. Training Coverage & Padding", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$w = {w}$" for w in windows])
    ax1.set_ylim(0, 24000)
    ax1.legend(loc="upper right", framealpha=0.9)

    for bar, val in zip(b1, kept):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 350, f"{val:,}\n({val/20631:.1%})",
                 ha="center", va="bottom", fontsize=8.5)
    for bar, val, test_pad in zip(b2, padded, cov_df["test_engines_padded"]):
        test_txt = f"\n[Test pad: {test_pad}]" if test_pad > 0 else "\n[Test pad: 0]"
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 350, f"{val:,}{test_txt}",
                 ha="center", va="bottom", fontsize=8.5, color="#b30000" if test_pad > 0 else "#333333")

    ax1.axhline(20631, color="gray", linestyle=":", alpha=0.7)
    ax1.text(0.02, 0.95, "Total train cycles = 20,631\nShortest test engine = 31 cycles",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox={"boxstyle": "round,pad=0.3", "fc": "#f7f7f7", "ec": "#cccccc", "alpha": 0.9})

    # 2. Right Plot: RUL Distribution Before and After Cap at 125
    ax2 = axes[1]
    bins = np.arange(0, 370, 10)
    ax2.hist(raw_df["rul"], bins=bins, color="#7570b3", alpha=0.55, edgecolor="black", linewidth=0.5,
             label=f"Raw RUL (Mean: {cap_stats['raw_mean']:.1f}, Max: {cap_stats['raw_max']:.0f})")
    ax2.hist(capped_df["rul"], bins=bins, color="#1b9e77", alpha=0.6, edgecolor="black", linewidth=0.5,
             label=f"Capped at 125 (Mean: {cap_stats['capped_mean']:.1f}, {cap_stats['pct_capped']:.1f}% at 125)")

    ax2.axvline(125, color="#d95f02", linestyle="--", linewidth=1.5, label="Cap threshold = 125 cycles")
    ax2.set_xlabel("Remaining Useful Life (cycles)", fontweight="bold")
    ax2.set_ylabel("Number of Cycles / Windows", fontweight="bold")
    ax2.set_title("(b) Target RUL Distribution (Before vs. After Cap)", fontweight="bold", pad=10)
    ax2.set_xlim(0, 370)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    ax2.annotate(
        f"Plateau spike at 125:\n{cap_stats['n_capped']:,} cycles ({cap_stats['pct_capped']:.1f}%)\nrepresent healthy operation",
        xy=(125, 5500), xytext=(160, 4200),
        arrowprops={"facecolor": "#d95f02", "shrink": 0.08, "width": 1.2, "headwidth": 6},
        fontsize=8.5, bbox={"boxstyle": "round,pad=0.3", "fc": "#fff2e6", "ec": "#d95f02", "alpha": 0.9},
    )

    plt.tight_layout()
    out_path = FIGURES / "ablation_coverage_padding.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {Path(out_path).relative_to(ROOT)}")


if __name__ == "__main__":
    plot_dataset_diagnostics()
