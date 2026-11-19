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
3. **Evidence Criterion:** Differences smaller than fold standard deviation ($\\sigma \\approx 0.35-0.50$) represent sampling variance rather than statistically significant effects.
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
    "ablation", "changed_factor", "n_params", "rmse_mean", "rmse_std",
    "mae_mean", "rmse_late_mean", "takeaway"
]])
"""))

    # Plot comparison code
    cells.append(nbf.v4.new_markdown_cell("""### Visualising Ablation Performance vs. Baseline
The forest plot below illustrates 5-fold CV RMSE with $\\pm 1\\sigma$ fold standard deviation error bars.
The blue dashed line and shaded area denote the baseline Transformer encoder ($13.28 \\pm 0.34$).
"""))

    cells.append(nbf.v4.new_code_cell("""from scripts.plot_ablations import plot_ablations_comparison

plot_ablations_comparison()
"""))

    # Viva discussion markdown
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Analysis & Viva Preparation

### Key Physical & Methodological Insights
1. **Window Length Trade-off ($w=15$ vs. $w=30$ vs. $w=50$):**
   - At $w=15$, CV RMSE rises sharply to $16.96$ ($+3.68$) and late RMSE jumps to $14.18$ ($+6.20$). Truncating history to 15 cycles deprives self-attention of the temporal curvature necessary to distinguish accelerated wear from noise.
   - At $w=50$, longer historical context yields strong in-sample fit on long engines, but forces 4,900 training cycles to be padded and, crucially, **7 test engines ($7\%$) must be artificially padded at test scoring** because their total life is shorter than 50 cycles.
   - **$w=30$ is the maximal window size guaranteeing 100% genuine test observations without padding.**

2. **Target Capping (125 vs. Uncapped):**
   - Training on uncapped targets forces the model to predict large differences in remaining life across healthy engines whose sensors show identical, un-degraded baselines.

3. **Regularisation Mechanics:**
   - Training on 64 engines per fold makes neural networks prone to overfitting. Early stopping acts as the primary regularizer, preventing divergence after validation loss bottoms out around epoch 10-16. Dropout ($0.1$) and weight decay ($10^{-4}$) prevent attention weight inflation and head co-adaptation.

4. **Sensor Subsets (Top SHAP vs. 14 Sensors):**
   - Restricting inputs to the top 3 SHAP sensors ($s_4, s_{15}, s_2$) confirms that 3 sensors capture the primary wear trajectory but lack the redundancy needed for fine-grained late predictions. Expanding to top 5 sensors ($+ s_{14}, s_{11}$) largely recovers accuracy.

5. **Hybrid Architecture Decomposition:**
   - The sequence branch provides the foundational temporal representations. While the Two-Branch Hybrid achieves the best late-stage RMSE ($7.53$), its additional MLP parameters introduce slight variance on early healthy cycles, making the standalone Transformer the overall RMSE champion.

### Viva: You Must Be Able to Explain
- **Why an ablation changes one factor at a time:** Changing multiple factors simultaneously creates confounding interactions; changing one factor isolates the precise causal contribution of each component.
- **Why a gap smaller than the fold spread is not evidence:** If the difference between two models is smaller than the fold standard deviation across folds, the observed margin cannot be distinguished from random engine sampling variance.
- **Why window length trades history against padding:** A longer window captures extended degradation trajectories, but increases the fraction of early cycles padded with initial readings and forces short test engines to be artificially padded at inference time.
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    make_notebook()
