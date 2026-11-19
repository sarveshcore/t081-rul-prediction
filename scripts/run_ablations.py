# ABOUTME: Script to execute all ablation cross-validations for Issue #37.
# ABOUTME: Saves incremental results to results/ablations.csv.
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rul.ablations import (
    FeatureMLPNet,
    SequenceMLPNet,
    feature_prepare,
)
from rul.data import USEFUL_SENSORS, add_rul, load_train_fd
from rul.evaluate import evaluate
from rul.nets import build_net, count_params
from rul.sequences import window_prepare
from rul.split import engine_folds, engine_split
from rul.train import SEED, cv_net, fit_net, predict_net, set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ABLATIONS_CSV = RESULTS_DIR / "ablations.csv"

FIT = {
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 100,
    "patience": 10,
    "grad_clip": 1.0,
    "y_scale": 125.0,
}

train_capped = add_rul(load_train_fd("FD001"), cap=125)
train_uncapped = add_rul(load_train_fd("FD001"), cap=None)

TOP3_SENSORS = ["s_4", "s_15", "s_2"]
TOP5_SENSORS = ["s_4", "s_15", "s_2", "s_14", "s_11"]


def run_uncapped_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on uncapped RUL; score held-out folds two ways.

    Returns (clipped, uncapped): predictions clipped at 125 and scored against the capped labels,
    which is like for like with every other row, and raw predictions scored against the true
    uncapped RUL, which is the model's own target but not comparable with the other rows.
    """
    rows, rows_uncapped = [], []
    prep = window_prepare(USEFUL_SENSORS, window=30)
    for fold, (train_idx, val_idx) in enumerate(engine_folds(train_capped, n_splits=5)):
        inner = train_uncapped.iloc[train_idx]
        inner_tr, es = engine_split(inner, val_frac=0.2, seed=SEED)

        X_tr, y_tr = prep(inner_tr, inner_tr)
        X_es, y_es = prep(inner_tr, es)
        # Validation inputs from inner scaling, targets evaluated against capped RUL
        val_df = train_capped.iloc[val_idx]
        X_val, y_val = prep(inner_tr, val_df)
        _, y_val_uncapped = prep(inner_tr, train_uncapped.iloc[val_idx])

        set_seed(SEED)
        net = build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
        history = fit_net(
            net,
            X_tr,
            y_tr,
            X_es,
            y_es,
            lr=1e-3,
            seed=SEED,
            weight_decay=1e-4,
            batch_size=256,
            epochs=100,
            patience=10,
            grad_clip=1.0,
            y_scale=125.0,
        )
        preds = predict_net(net, X_val, y_scale=125.0)
        common = {"best_epoch": history["best_epoch"], "n_rows": len(y_val)}
        rows.append({"fold": fold, **evaluate(y_val, np.clip(preds, 0.0, 125.0)), **common})
        rows_uncapped.append({"fold": fold, **evaluate(y_val_uncapped, preds), **common})
    return pd.DataFrame(rows), pd.DataFrame(rows_uncapped)


def run_no_es_ablation(dropout: float = 0.1, weight_decay: float = 1e-4, epochs: int = 40) -> pd.DataFrame:
    """Train for a fixed number of epochs and keep the last epoch's weights: no early stopping.

    fit_net restores the best early stopping epoch by default, so restore_best=False is what
    removes early stopping; patience=epochs alone would still select the best epoch.
    """
    rows = []
    prep = window_prepare(USEFUL_SENSORS, window=30)
    for fold, (train_idx, val_idx) in enumerate(engine_folds(train_capped, n_splits=5)):
        inner, es = engine_split(train_capped.iloc[train_idx], val_frac=0.2, seed=SEED)
        X_tr, y_tr = prep(inner, inner)
        X_es, y_es = prep(inner, es)
        val_df = train_capped.iloc[val_idx]
        X_val, y_val = prep(inner, val_df)

        set_seed(SEED)
        net = build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=dropout)
        history = fit_net(
            net,
            X_tr,
            y_tr,
            X_es,
            y_es,
            lr=1e-3,
            seed=SEED,
            weight_decay=weight_decay,
            batch_size=256,
            epochs=epochs,
            patience=epochs,  # never stop before the last epoch
            grad_clip=1.0,
            y_scale=125.0,
            restore_best=False,  # and keep the last epoch's weights
        )
        preds = predict_net(net, X_val, y_scale=125.0)
        metrics = evaluate(y_val, preds)
        rows.append({
            "fold": fold,
            **metrics,
            "best_epoch": history["best_epoch"],
            "n_rows": len(y_val),
        })
    return pd.DataFrame(rows)


WINDOW_UNPADDED_CSV = RESULTS_DIR / "ablation_window_unpadded.csv"


def run_window_unpadded(windows: tuple[int, ...] = (30, 50), min_cycle: int = 50) -> pd.DataFrame:
    """CV RMSE of the baseline Transformer per window length, on all cycles and on cycles >= min_cycle.

    No window ending at cycle >= min_cycle is padded for any length up to min_cycle, so the second
    score shows whether a longer window helps through real history rather than through padding
    revealing how early in life an engine is.
    """
    rows = []
    for w in windows:
        prep = window_prepare(USEFUL_SENSORS, window=w)
        for fold, (train_idx, val_idx) in enumerate(engine_folds(train_capped, n_splits=5)):
            inner, es = engine_split(train_capped.iloc[train_idx], val_frac=0.2, seed=SEED)
            X_tr, y_tr = prep(inner, inner)
            X_es, y_es = prep(inner, es)
            val_df = train_capped.iloc[val_idx]
            X_val, y_val = prep(inner, val_df)
            cycles = val_df.sort_values(["unit", "cycle"])["cycle"].to_numpy()
            set_seed(SEED)
            net = build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
            fit_net(net, X_tr, y_tr, X_es, y_es, lr=1e-3, seed=SEED, **FIT)
            err = predict_net(net, X_val, y_scale=125.0) - y_val
            keep = cycles >= min_cycle
            rows.append({"window": w, "fold": fold,
                         "rmse_all_cycles": float(np.sqrt(np.mean(err ** 2))),
                         f"rmse_cycles_ge_{min_cycle}": float(np.sqrt(np.mean(err[keep] ** 2))),
                         f"n_rows_ge_{min_cycle}": int(keep.sum())})
    return pd.DataFrame(rows)


def format_row(name, factor, net_builder, folds_df, takeaway):
    net = net_builder() if callable(net_builder) else None
    params = count_params(net) if net is not None else None
    return {
        "ablation": name,
        "changed_factor": factor,
        "n_params": params,
        "rmse_mean": float(folds_df["rmse"].mean()),
        "rmse_std": float(folds_df["rmse"].std()),
        "mae_mean": float(folds_df["mae"].mean()),
        "mae_std": float(folds_df["mae"].std()),
        "nasa_score_mean": float(folds_df["nasa_score"].mean()),
        "rmse_late_mean": float(folds_df["rmse_late"].mean()),
        "rmse_late_std": float(folds_df["rmse_late"].std()),
        "takeaway": takeaway,
    }


def main():
    print("=== Running Ablation Studies (Issue #37) ===")
    results = []

    # 1. Baseline Transformer, rerun here with the same cv_net call as every other row, so each
    # ablation's difference from it is like for like (dl_cv.csv comes from a separate run).
    print("--- Baseline Transformer ---")
    t0 = time.time()
    prep30 = window_prepare(USEFUL_SENSORS, window=30)
    build_t30 = lambda: build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
    fbase = cv_net(build_t30, train_capped, prep30, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Baseline (Transformer)",
        "None (reference: w=30, cap 125, dropout 0.1, weight decay 1e-4, early stopping, 14 sensors)",
        build_t30,
        fbase,
        "Reference for every row below, rerun in this script.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 2. Window length 15
    print("\n--- Ablation: Window length w=15 ---")
    t0 = time.time()
    prep15 = window_prepare(USEFUL_SENSORS, window=15)
    build_t15 = lambda: build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
    f15 = cv_net(build_t15, train_capped, prep15, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Window length: w=15",
        "window = 15 (reduced from 30)",
        build_t15,
        f15,
        "Much worse overall and near failure: 15 cycles of history is too little.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 3. Window length 50
    print("\n--- Ablation: Window length w=50 ---")
    t0 = time.time()
    prep50 = window_prepare(USEFUL_SENSORS, window=50)
    build_t50 = lambda: build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
    f50 = cv_net(build_t50, train_capped, prep50, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Window length: w=50",
        "window = 50 (increased from 30)",
        build_t50,
        f50,
        "Clearly better on CV, on every fold and also on unpadded cycles >= 50; not adopted because the test set was already opened (#48).",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 4. Target: No RUL cap
    print("\n--- Ablation: Target: No RUL cap ---")
    t0 = time.time()
    funcap, funcap_true = run_uncapped_ablation()
    build_uncap = lambda: build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.1)
    results.append(format_row(
        "Target: No RUL cap",
        "cap = None (trained on uncapped RUL; predictions clipped at 125, scored against capped RUL)",
        build_uncap,
        funcap,
        "Training without the cap is worse even when its predictions are clipped to the same scale as every other row.",
    ))
    results.append(format_row(
        "Target: No RUL cap, scored on its own target",
        "same model, raw predictions scored against true uncapped RUL (not comparable with other rows)",
        build_uncap,
        funcap_true,
        "Its own target is harder: early cycles of healthy engines carry RUL up to 361 that the sensors cannot show.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-2]['rmse_mean']:.3f} (clipped) {results[-1]['rmse_mean']:.3f} (uncapped)")

    # 5. Regularisation: No dropout
    print("\n--- Ablation: Reg: No dropout ---")
    t0 = time.time()
    build_nodrop = lambda: build_net("transformer", len(USEFUL_SENSORS), d_model=64, layers=1, dropout=0.0)
    fnodrop = cv_net(build_nodrop, train_capped, prep30, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Regularisation: No dropout",
        "dropout = 0.0 (down from 0.1)",
        build_nodrop,
        fnodrop,
        "Within the fold spread of the baseline: no measurable effect.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 6. Regularisation: No weight decay
    print("\n--- Ablation: Reg: No weight decay ---")
    t0 = time.time()
    fit_nowd = dict(FIT)
    fit_nowd["weight_decay"] = 0.0
    fnowd = cv_net(build_t30, train_capped, prep30, lr=1e-3, seed=SEED, **fit_nowd)
    results.append(format_row(
        "Regularisation: No weight decay",
        "weight_decay = 0.0 (down from 1e-4)",
        build_t30,
        fnowd,
        "Within the fold spread of the baseline: no measurable effect.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 7. Regularisation: No early stopping
    print("\n--- Ablation: Reg: No early stopping ---")
    t0 = time.time()
    fnoes = run_no_es_ablation(dropout=0.1, weight_decay=1e-4, epochs=40)
    results.append(format_row(
        "Regularisation: No early stopping",
        "no early stopping: 40 epochs, last epoch's weights kept",
        build_t30,
        fnoes,
        "Keeping the last of 40 epochs instead of the best early stopping epoch.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 8. Regularisation: All off
    print("\n--- Ablation: Reg: All regularisation off ---")
    t0 = time.time()
    falloff = run_no_es_ablation(dropout=0.0, weight_decay=0.0, epochs=40)
    results.append(format_row(
        "Regularisation: All off",
        "dropout = 0.0, weight_decay = 0.0, no early stopping (40 epochs, last weights)",
        build_nodrop,
        falloff,
        "All three regularisers removed together.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 9. Inputs: Top 3 SHAP sensors
    print("\n--- Ablation: Inputs: Top 3 SHAP sensors ---")
    t0 = time.time()
    prep_top3 = window_prepare(TOP3_SENSORS, window=30)
    build_top3 = lambda: build_net("transformer", len(TOP3_SENSORS), d_model=64, layers=1, dropout=0.1)
    ftop3 = cv_net(build_top3, train_capped, prep_top3, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Inputs: Top 3 SHAP sensors",
        "sensors = [s_4, s_15, s_2] (down from 14)",
        build_top3,
        ftop3,
        "Much worse overall and near failure: three sensors are not enough.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 10. Inputs: Top 5 SHAP sensors
    print("\n--- Ablation: Inputs: Top 5 SHAP sensors ---")
    t0 = time.time()
    prep_top5 = window_prepare(TOP5_SENSORS, window=30)
    build_top5 = lambda: build_net("transformer", len(TOP5_SENSORS), d_model=64, layers=1, dropout=0.1)
    ftop5 = cv_net(build_top5, train_capped, prep_top5, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Inputs: Top 5 SHAP sensors",
        "sensors = [s_4, s_15, s_2, s_14, s_11] (down from 14)",
        build_top5,
        ftop5,
        "Recovers part of the gap from top 3 but stays well behind all 14 sensors.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 11. Hybrid: Sequence branch only
    print("\n--- Ablation: Hybrid: Sequence branch only ---")
    t0 = time.time()
    build_seq_mlp = lambda: SequenceMLPNet(n_features=len(USEFUL_SENSORS), d_model=64, layers=1, heads=4, mlp_hidden=64, encoder_dropout=0.1, head_dropout=0.2)
    fseq_mlp = cv_net(build_seq_mlp, train_capped, prep30, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Hybrid: Sequence branch only",
        "HybridNet with 0 engineered features (Sequence + MLP head)",
        build_seq_mlp,
        fseq_mlp,
        "Matches the plain Transformer within the fold spread; lower late RMSE than the two-branch hybrid.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 12. Hybrid: Feature branch only
    print("\n--- Ablation: Hybrid: Feature branch only ---")
    t0 = time.time()
    build_feat_mlp = lambda: FeatureMLPNet(n_features=142, mlp_hidden=64, dropout=0.2)
    prep_feat = feature_prepare(USEFUL_SENSORS)
    ffeat_mlp = cv_net(build_feat_mlp, train_capped, prep_feat, lr=1e-3, seed=SEED, **FIT)
    results.append(format_row(
        "Hybrid: Feature branch only",
        "HybridNet with no sequence window (142 features + MLP head)",
        build_feat_mlp,
        ffeat_mlp,
        "Far behind the Transformer and LSTM: the sensor window carries most of the signal.",
    ))
    pd.DataFrame(results).to_csv(ABLATIONS_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s | RMSE: {results[-1]['rmse_mean']:.3f} +- {results[-1]['rmse_std']:.3f}")

    # 13. Two-Branch Hybrid Reference (from results/hybrid_cv.csv)
    hybrid_cv = pd.read_csv(RESULTS_DIR / "hybrid_cv.csv")
    hyb_row = hybrid_cv[hybrid_cv["model"] == "Hybrid (Transformer + Engineered)"].iloc[0]
    results.append({
        "ablation": "Hybrid: Two-Branch (Seq + Feat)",
        "changed_factor": "Both branches combined (30-cycle window + 142 features)",
        "n_params": int(hyb_row["n_params"]),
        "rmse_mean": float(hyb_row["rmse_mean"]),
        "rmse_std": float(hyb_row["rmse_std"]),
        "mae_mean": float(hyb_row["mae_mean"]),
        "mae_std": float(hyb_row["mae_std"]),
        "nasa_score_mean": float(hyb_row["nasa_score_mean"]),
        "rmse_late_mean": float(hyb_row["rmse_late_mean"]),
        "rmse_late_std": float(hyb_row["rmse_late_std"]),
        "takeaway": "Both branches: no better than the sequence branch alone, overall or near failure.",
    })

    final_df = pd.DataFrame(results)
    # One run, so every row is compared with the baseline computed above in the same process.
    # Torch thread count changes the training trajectory, so rows from separate runs are only
    # compared with a baseline from their own run (see the run and reference columns).
    final_df["run"] = f"{torch.get_num_threads()} torch threads"
    final_df["reference"] = "Baseline (Transformer)"
    final_df.loc[final_df["changed_factor"].str.contains("not comparable"), "reference"] = None
    base = final_df.loc[final_df["ablation"] == "Baseline (Transformer)", "rmse_mean"].iloc[0]
    final_df["rmse_delta"] = (final_df["rmse_mean"] - base).where(final_df["reference"].notna())
    final_df.to_csv(ABLATIONS_CSV, index=False)

    print("\n--- Window check on unpadded cycles (w=30 vs w=50, cycles >= 50) ---")
    run_window_unpadded().to_csv(WINDOW_UNPADDED_CSV, index=False)
    print("\n=== All Ablations Complete! Saved to results/ablations.csv ===")
    print(final_df[["ablation", "n_params", "rmse_mean", "rmse_std", "mae_mean", "rmse_late_mean"]])


if __name__ == "__main__":
    main()
