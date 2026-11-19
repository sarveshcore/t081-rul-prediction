# ABOUTME: Tests for hybrid architectures combining sequence encoders and engineered features.
# ABOUTME: Verifies window-feature alignment, zero future cycle leakage, and scaler isolation.
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from rul.features import FeatureScaler
from rul.hybrid import HybridNet, cv_residual_hybrid, engineered_columns, hybrid_prepare
from rul.nets import count_params
from rul.train import fit_net, predict_net, set_seed


class ToyEncoder(nn.Module):
    """Minimal sequence encoder returning a fixed-width vector per window."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(2, out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[:, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[:, -1]).mean(dim=-1)


def toy_engines(seed: int = 42) -> pd.DataFrame:
    """Synthetic engines of varying lengths with two useful sensors and settings."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, 7):
        n_cycles = 15 + u * 3
        for c in range(1, n_cycles + 1):
            rows.append({
                "unit": u,
                "cycle": c,
                "setting_1": rng.normal(0, 0.002),
                "setting_2": rng.normal(0, 0.0003),
                "setting_3": 100.0,
                "s_2": 640.0 + 0.02 * c + rng.normal(0, 0.1),
                "s_3": 1590.0 + 0.15 * c + rng.normal(0, 0.5),
                "rul": float(n_cycles - c),
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


def test_hybrid_net_output_shape():
    encoder = ToyEncoder(out_dim=16)
    net = HybridNet(encoder, n_features=20, hidden=32, dropout=0.1).eval()
    windows = torch.randn(8, 10, 2)
    features = torch.randn(8, 20)
    out = net(windows, features)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_hybrid_net_parameter_count():
    encoder = ToyEncoder(out_dim=16)
    net = HybridNet(encoder, n_features=10, hidden=32, dropout=0.1)
    # encoder: Linear(2, 16) -> 2*16 + 16 = 48
    # head: Linear(16+10, 32) -> 26*32 + 32 = 864; Linear(32, 1) -> 33. Total head = 897.
    # Total = 945
    assert count_params(net) == 48 + 897


def test_hybrid_net_drops_unused_encoder_head():
    """If the encoder carries its own head, HybridNet replaces it with Identity so dead weights are dropped."""
    class EncoderWithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = 16
            self.linear = nn.Linear(2, 16)
            self.head = nn.Linear(16, 1)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x[:, -1])

    enc = EncoderWithHead()
    net = HybridNet(enc, n_features=10, hidden=32, dropout=0.1)
    assert isinstance(enc.head, nn.Identity)
    # The head's 17 parameters (16*1 + 1) are replaced by Identity (0 params)
    assert count_params(net) == 48 + 897


def test_hybrid_prepare_aligns_window_and_features():
    """Row k of features must correspond to the exact engine and cycle where window k ends."""
    df = toy_engines()
    sensors = ["s_2", "s_3"]
    window_len = 5
    cols = engineered_columns(sensors, (3, 5))
    prepare = hybrid_prepare(sensors, window=window_len, windows_tuple=(3, 5))
    (X_seq, X_feat), y = prepare(df, df)

    sorted_df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    assert len(X_seq) == len(X_feat) == len(sorted_df) == len(y)

    # 1. Target check
    np.testing.assert_allclose(y, sorted_df["rul"].to_numpy())

    # 2. Window terminal cycle check: last timestep of window k must match sensor values of row k
    seq_scaler = StandardScaler().fit(sorted_df[sensors])
    unscaled_last_step = seq_scaler.inverse_transform(X_seq[:, -1, :])
    np.testing.assert_allclose(unscaled_last_step, sorted_df[sensors].to_numpy(), rtol=1e-5, atol=1e-5)

    # 3. Feature check: cycle column and raw sensor columns must match row k
    feat_scaler = FeatureScaler.fit(sorted_df, sensors, (3, 5))
    fit_feat = feat_scaler.transform(sorted_df)
    feat_norm = StandardScaler().fit(fit_feat[cols])
    unscaled_feat = feat_norm.inverse_transform(X_feat)

    cycle_idx = cols.index("cycle")
    np.testing.assert_allclose(unscaled_feat[:, cycle_idx], sorted_df["cycle"].to_numpy(), rtol=1e-5, atol=1e-5)
    for s in sensors:
        s_idx = cols.index(s)
        np.testing.assert_allclose(unscaled_feat[:, s_idx], sorted_df[s].to_numpy(), rtol=1e-5, atol=1e-5)


def test_hybrid_prepare_fits_on_fit_df_only():
    """Features on apply_df must not shift if apply_df target or distributions are modified."""
    fit_df = toy_engines(seed=42)
    apply_df = toy_engines(seed=99)
    sensors = ["s_2", "s_3"]

    prepare = hybrid_prepare(sensors, window=5, windows_tuple=(3, 5))
    (X_seq1, X_feat1), y1 = prepare(fit_df, apply_df)

    # 1. Shifting apply_df RUL labels must not change feature representations
    apply_modified = apply_df.copy()
    apply_modified["rul"] += 50.0
    (X_seq2, X_feat2), y2 = prepare(fit_df, apply_modified)

    np.testing.assert_array_equal(X_seq1, X_seq2)
    np.testing.assert_array_equal(X_feat1, X_feat2)
    np.testing.assert_array_equal(y2, y1 + 50.0)

    # 2. Shifting sensor values in apply_df uses fit_df's scaling parameters,
    # proving that scalers do not fit on apply_df
    apply_sensor_shift = apply_df.copy()
    shift_val = 1000.0
    apply_sensor_shift["s_2"] += shift_val

    (X_seq_shift, X_feat_shift), _ = prepare(fit_df, apply_sensor_shift)

    s2_std_fit = fit_df["s_2"].std(ddof=0)
    expected_shift = shift_val / s2_std_fit

    # Sequence window values for s_2 must shift by exactly expected_shift
    np.testing.assert_allclose(X_seq_shift[:, :, 0], X_seq1[:, :, 0] + expected_shift, rtol=1e-5)
    # Unshifted sensor s_3 in sequence window must be completely unaffected
    np.testing.assert_array_equal(X_seq_shift[:, :, 1], X_seq1[:, :, 1])
    # Feature matrix shape matches and incorporates sensor shift
    assert X_feat_shift.shape == X_feat1.shape
    assert not np.array_equal(X_feat_shift, X_feat1)


def test_no_future_cycle_leakage_in_features():
    """Modifying future sensor readings of an engine must not change features of earlier cycles."""
    df = toy_engines(seed=42)
    sensors = ["s_2", "s_3"]
    prepare = hybrid_prepare(sensors, window=5, windows_tuple=(3, 5))

    (X_seq_orig, X_feat_orig), _ = prepare(df, df)

    # Pick unit 1, corrupt its late cycles (> 10)
    corrupted = df.copy()
    mask = (corrupted["unit"] == 1) & (corrupted["cycle"] > 10)
    corrupted.loc[mask, "s_2"] += 500.0
    corrupted.loc[mask, "s_3"] -= 500.0

    (X_seq_corrupt, X_feat_corrupt), _ = prepare(df, corrupted)

    sorted_orig = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    early_unit1_indices = sorted_orig[(sorted_orig["unit"] == 1) & (sorted_orig["cycle"] <= 10)].index

    # Early cycle features and windows must be 100% identical
    np.testing.assert_allclose(X_feat_orig[early_unit1_indices], X_feat_corrupt[early_unit1_indices], atol=1e-5)
    np.testing.assert_allclose(X_seq_orig[early_unit1_indices], X_seq_corrupt[early_unit1_indices], atol=1e-5)


def test_hybrid_net_trains_and_learns():
    """Verifies that fit_net and predict_net work end-to-end with HybridNet."""
    set_seed(42)
    encoder = ToyEncoder(out_dim=8)
    net = HybridNet(encoder, n_features=6, hidden=16, dropout=0.0)

    windows = torch.randn(64, 10, 2)
    features = torch.randn(64, 6)
    # Synthetic target: combination of window and feature signals
    target = 50.0 + 10.0 * windows[:, -1, 0] + 5.0 * features[:, 0]

    history = fit_net(net, (windows, features), target, (windows, features), target,
                      epochs=30, lr=1e-2, batch_size=32, y_scale=125.0)
    assert history["val_loss"][-1] < history["val_loss"][0]

    pred = predict_net(net, (windows, features), y_scale=125.0)
    assert pred.shape == (64,)
    assert np.isfinite(pred).all()


def test_cv_residual_hybrid_runs():
    """Smoke test ensuring cv_residual_hybrid runs with small folds and reasonable metrics."""
    df = toy_engines(seed=42)
    sensors = ["s_2", "s_3"]

    def build_toy():
        return ToyEncoder(out_dim=8)

    xgb_fast = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
    }

    results = cv_residual_hybrid(
        build_toy,
        df,
        sensors=sensors,
        window=5,
        windows_tuple=(3, 5),
        xgb_params=xgb_fast,
        n_splits=2,
        epochs=3,
        batch_size=32,
    )
    assert len(results) == 2
    assert "rmse" in results.columns
    assert np.isfinite(results["rmse"].to_numpy()).all()
