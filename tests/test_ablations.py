# ABOUTME: Unit tests for dataset coverage diagnostics and ablation helper models.
# ABOUTME: Validates window coverage arithmetic, padding counts, RUL capping, and ablation network shapes.
import numpy as np
import pytest
import torch

from rul.ablations import (
    FeatureMLPNet,
    SequenceMLPNet,
    cap_distribution,
    dataset_coverage,
    feature_prepare,
    regime_variation,
)
from rul.data import USEFUL_SENSORS, add_rul, load_train_fd
from rul.nets import count_params


def test_dataset_coverage_counts():
    cov = dataset_coverage((15, 30, 50))
    assert len(cov) == 3
    assert list(cov["window"]) == [15, 30, 50]

    # Total cycles and engines
    assert (cov["train_cycles_total"] == 20631).all()
    assert (cov["test_engines_total"] == 100).all()
    assert (cov["shortest_test_engine"] == 31).all()

    # Exact training counts
    assert list(cov["train_cycles_kept"]) == [19231, 17731, 15731]
    assert list(cov["train_windows_padded"]) == [1400, 2900, 4900]
    assert cov["train_cycles_kept"].iloc[0] > cov["train_cycles_kept"].iloc[1] > cov["train_cycles_kept"].iloc[2]

    # Test engine padding: 0 padded for w <= 31, exactly 7 padded for w = 50
    assert cov.loc[cov["window"] == 15, "test_engines_padded"].iloc[0] == 0
    assert cov.loc[cov["window"] == 30, "test_engines_padded"].iloc[0] == 0
    assert cov.loc[cov["window"] == 50, "test_engines_padded"].iloc[0] == 7
    assert cov.loc[cov["window"] == 50, "test_engines_kept"].iloc[0] == 93


def test_cap_distribution_statistics():
    stats = cap_distribution(cap=125)

    assert stats["raw_max"] == 361.0
    assert stats["capped_max"] == 125.0
    assert stats["n_capped"] == 8031
    assert stats["pct_capped"] == pytest.approx(38.9268, abs=0.01)

    assert stats["raw_mean"] == pytest.approx(107.8078, abs=0.01)
    assert stats["capped_mean"] == pytest.approx(86.8292, abs=0.01)
    assert stats["raw_std"] == pytest.approx(68.8809, abs=0.01)
    assert stats["capped_std"] == pytest.approx(41.6736, abs=0.01)


def test_regime_variation_verifies_single_condition():
    reg = regime_variation()
    assert len(reg) == 3
    s1 = reg[reg["setting"] == "setting_1"].iloc[0]
    s2 = reg[reg["setting"] == "setting_2"].iloc[0]
    s3 = reg[reg["setting"] == "setting_3"].iloc[0]

    assert s1["std"] < 0.005
    assert s2["std"] < 0.001
    assert s3["std"] == 0.0
    assert s3["min"] == s3["max"] == 100.0


def test_sequence_mlp_net_shape_and_params():
    net = SequenceMLPNet(n_features=14, d_model=64, layers=1, heads=4, mlp_hidden=64, encoder_dropout=0.1, head_dropout=0.2).eval()
    windows = torch.randn(4, 30, 14)
    out = net(windows)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()

    # Base encoder without head: 50,944 + MLP head: Linear(64, 64) -> 4160 + Linear(64, 1) -> 65 = 4225. Total = 55,169
    params = count_params(net)
    assert params > 50000


def test_feature_mlp_net_shape_and_params():
    net = FeatureMLPNet(n_features=142, mlp_hidden=64, dropout=0.2).eval()
    features = torch.randn(4, 142)
    out = net(features)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()

    # Linear(142, 64) -> 142*64 + 64 = 9152; Linear(64, 1) -> 65. Total = 9217
    assert count_params(net) == 9217


def test_feature_prepare_pipeline():
    train = add_rul(load_train_fd("FD001"), cap=125)
    subset = train[train["unit"] <= 5]
    prep = feature_prepare(USEFUL_SENSORS)
    X, y = prep(subset, subset)

    assert X.shape[0] == len(subset)
    assert X.shape[1] == 142
    assert y.shape[0] == len(subset)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
