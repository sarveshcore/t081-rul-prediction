# ABOUTME: Generates notebooks/08_ablations.ipynb following repository conventions.
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "08_ablations.ipynb"


def make_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Intro markdown
    cells.append(nbf.v4.new_markdown_cell("""# DL dataset choices and ablation studies

This notebook addresses two core rubric items on NASA C-MAPSS FD001:
1. **DL Dataset Choices and Justification:**
   - Evaluates sequence window length coverage ($w \\in \\{15, 30, 50\\}$) on training cycles and test engines.
   - Quantifies padding requirements (shortest test engine is 31 cycles).
   - Justifies the piece-wise linear target RUL cap at 125 cycles against raw uncapped RUL.
   - Verifies sequence operating regime normalisation against the single-regime property of FD001.
2. **Regularization and Ablation Studies:**
   - Single-factor-at-a-time ablations systematically removing or altering one component from the chosen sequence model (Transformer encoder, $w=30$, cap 125, dropout 0.1, weight decay $10^{-4}$, patience 10 on 20% nested validation, 14 useful sensors).
   - Evaluates window lengths ($w=15, 30, 50$), target capping, regularisation components (dropout, weight decay, early stopping), input sensors (all 14 vs. top SHAP sensors), and hybrid architectural branches.
   - Scored on the standard 5-fold engine-grouped cross-validation without test set leakage.
"""))

    # Setup code cell
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul.ablations import cap_distribution, dataset_coverage, regime_variation

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ABLATIONS_CSV = RESULTS / "ablations.csv"

LOAD_SAVED = True  # Set to False to re-run all 5-fold cross-validations from scratch (~15-20 min on CPU)
pd.set_option("display.precision", 3)
pd.set_option("display.max_colwidth", None)

print("Environment initialized.")
print(f"LOAD_SAVED = {LOAD_SAVED}")
"""))

    # Dataset coverage markdown
    cells.append(nbf.v4.new_markdown_cell("""## 1. Deep Learning Dataset Choices and Justification

### Window Length Coverage and Padding Trade-off
Sequence models map a temporal history of $w$ cycles of 14 sensor readings to remaining useful life (RUL).
Because an engine has only $t$ observations at cycle $t$, windows for cycles $t < w$ have fewer than $w$ historical cycles.
Rather than discard healthy early cycles, we left-pad each window by repeating that engine's initial cycle reading.
Since healthy sensors are nearly stationary, repeating cycle 1 introduces no false degradation slope and strictly avoids borrowing readings across engines.

However, window length introduces a direct physical and operational trade-off:
- Shortest training engine has 128 cycles.
- **Shortest test engine (unit 1) has only 31 cycles.**
"""))

    # Dataset coverage code
    cells.append(nbf.v4.new_code_cell("""cov_df = dataset_coverage((15, 30, 50))
cov_df[[
    "window", "train_cycles_total", "train_cycles_kept", "train_kept_pct",
    "train_windows_padded", "test_engines_total", "test_engines_kept",
    "test_engines_padded", "shortest_test_engine"
]]
"""))

    # Target cap & regime markdown
    cells.append(nbf.v4.new_markdown_cell("""### Target Capping and Operating Regime Equivalence
1. **Target RUL Cap at 125 cycles:** Before degradation begins, turbofan sensors exhibit stationary baselines identical between an engine with 300 cycles remaining and one with 150 cycles remaining. Forcing networks to regress uncapped RUL over healthy cycles penalizes them for unobservable variance and pulls gradients away from late wear. Clipping RUL at 125 cycles establishes a healthy plateau for 38.9% of cycles.
2. **Operating Regime Normalisation:** FD001 features constant sea-level operating settings with standard deviations $\\le 0.0022$. Standardizing sensors by operating regime is mathematically identical to global standardization across training engines.
"""))

    # Dataset stats code
    cells.append(nbf.v4.new_code_cell("""cap_stats = cap_distribution(125)
print("RUL Target Distribution (Cap = 125):")
for k, v in cap_stats.items():
    print(f"  {k:15s}: {v:10.2f}")

print("\\nOperating Setting Variability (FD001):")
display(regime_variation())
"""))

    # Figure display markdown & code
    cells.append(nbf.v4.new_markdown_cell("""### Visualising Dataset Coverage and Target Distribution
![Dataset Coverage and Padding Trade-off](../figures/ablation_coverage_padding.png)
"""))

    cells.append(nbf.v4.new_code_cell("""from scripts.plot_coverage import plot_dataset_diagnostics
 
plot_dataset_diagnostics()
"""))

    # Ablations markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Regularization and Ablation Studies

### Experimental Methodology
Following strict scientific ablation discipline:
1. **Single Factor at a Time:** Every ablation modifies *exactly one* component relative to the baseline Transformer sequence model ($w=30$, cap 125, dropout 0.1, weight decay $10^{-4}$, patience 10 on 20% nested validation, 14 useful sensors).
2. **Identical 5-Fold Engine Cross-Validation:** All runs use the shared engine folds from Phase 1, scoring every cycle against RUL capped at 125.
3. **Two runs, each with its own baseline:** the same baseline configuration gives CV RMSE 13.00 with default PyTorch threads and 13.17 with one thread (13.28 in `dl_cv.csv`), because the order of floating point operations changes training. The rows come from two runs, so each is compared only with the baseline of its own run (`run` and `reference` columns, `rmse_delta`).
4. **Evidence criterion:** a difference counts only if it is larger than the fold standard deviation of both rows and than the run to run variation of about 0.3 cycles.
5. **No early stopping** means 40 epochs keeping the last epoch's weights (`fit_net(..., restore_best=False)`); otherwise `fit_net` restores the best early stopping epoch.
"""))

    # Ablations execution or load code
    cells.append(nbf.v4.new_code_cell("""if LOAD_SAVED and ABLATIONS_CSV.exists():
    ablations_df = pd.read_csv(ABLATIONS_CSV)
    print(f"Loaded {len(ablations_df)} ablations from {ABLATIONS_CSV.relative_to(ROOT)}")
else:
    print("Running full ablation suite via scripts/run_ablations.py...")
    from scripts.run_ablations import main as run_all_ablations
    run_all_ablations()
    ablations_df = pd.read_csv(ABLATIONS_CSV)

display(ablations_df[[
    "ablation", "changed_factor", "run", "rmse_mean", "rmse_std", "rmse_delta",
    "mae_mean", "rmse_late_mean", "takeaway"
]])
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Is the w=50 gain only padding?
A padded window shows how early in life an engine is. Scoring only cycles >= 50, where no window of either length is padded, separates real history from that cue.
"""))
    cells.append(nbf.v4.new_code_cell("""window_check = pd.read_csv(RESULTS / "ablation_window_unpadded.csv")
display(window_check.pivot(index="fold", columns="window", values="rmse_cycles_ge_50"))
window_check.groupby("window")[["rmse_all_cycles", "rmse_cycles_ge_50"]].agg(["mean", "std"])
"""))

    # Plot comparison code
    cells.append(nbf.v4.new_markdown_cell("""### Visualising Ablation Performance vs. Baseline
Each bar is a row's CV RMSE minus the baseline of its own run; the error bar is that row's fold standard deviation.
"""))

    cells.append(nbf.v4.new_code_cell("""from scripts.plot_ablations import plot_ablations_comparison

plot_ablations_comparison()
"""))

    # Viva discussion markdown
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Findings

1. **Window length.** $w=15$ is clearly worse ($+3.96$). $w=50$ is clearly better ($-1.66$, 2.5 to 3 fold standard deviations), and stays better on cycles $\ge 50$ where nothing is padded (11.96 vs 14.19, every fold), so longer history genuinely helps on CV. We keep $w=30$ only because the window was fixed before the test set was opened in #48; switching now would be a choice made after seeing test results. It is reported as a limitation.
2. **Target cap.** Without the cap, even with predictions clipped at 125, RMSE is 2.25 worse and the fold spread doubles. On its own uncapped target the RMSE is 38.57.
3. **Regularisation.** No dropout ($+0.54$) and no weight decay ($+0.46$) are within the fold spread: no measurable effect. No early stopping is clearly worse ($+2.48$), and all three off ($+2.83$) is about the same, so early stopping is the regulariser that matters.
4. **Inputs.** Top 3 SHAP sensors: $+5.70$. Top 5: $+2.44$, still well behind all 14.
5. **Hybrid branches.** Sequence branch alone $\approx$ plain Transformer ($+0.33$). Feature branch alone $+2.29$. Both branches are no better than the sequence branch alone, overall or near failure (late 7.53 vs 7.34).

### Viva: you must be able to explain
- **Why an ablation changes one factor at a time:** with two changes you cannot tell which one caused the difference.
- **Why a gap smaller than the fold spread is not evidence:** a different split of engines into folds could move it by that much.
- **Why each row needs a baseline from its own run:** the thread count alone moves the baseline from 13.00 to 13.17.
- **Why window length trades history against padding:** longer windows see more history but pad more early cycles, and at $w=50$ seven test engines are shorter than the window.
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    make_notebook()
