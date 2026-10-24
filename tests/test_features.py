# ABOUTME: Tests for degradation features in rul.features.
# ABOUTME: Guards against leakage: windows stay inside one unit, scalers fit on train only.
import numpy as np
import pandas as pd
import pytest

from rul.features import feature_groups, make_features

SENSORS = ["s_2", "s_3", "s_7"]
WINDOWS = (5, 10, 20)


def _engines(units: list[int], seed: int = 42) -> pd.DataFrame:
    """Synthetic engines of uneven length, rows shuffled, non default index."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in units:
        for c in range(1, 8 + 3 * u):
            rows.append({
                "unit": u, "cycle": c,
                "setting_1": rng.normal(0, 0.002), "setting_2": rng.normal(0, 0.0003),
                "setting_3": 100.0,
                "s_2": 640 + 0.01 * c + rng.normal(0, 0.1),
                "s_3": 1590 + 0.1 * c + rng.normal(0, 1),
                "s_7": 555 - 0.02 * c + rng.normal(0, 0.2),
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed)
    return df.set_axis(df.index + 1000)


@pytest.fixture
def train():
    """Six training engines."""
    return _engines([1, 2, 3, 4, 5, 6])


def test_output_has_every_documented_column(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    for columns in feature_groups(SENSORS, WINDOWS).values():
        assert set(columns) <= set(out.columns)


def test_rows_and_index_preserved(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    assert len(out) == len(train)
    assert out.index.equals(train.index)
    pd.testing.assert_frame_equal(out[train.columns], train)


def test_no_nan_even_at_first_cycle(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    assert not out.isna().any().any()
    first = out[out.cycle == 1]
    slopes = feature_groups(SENSORS, WINDOWS)["slope"]
    stds = [c for c in feature_groups(SENSORS, WINDOWS)["rolling"] if "_std_" in c]
    assert (first[slopes] == 0).all().all()
    assert (first[stds] == 0).all().all()


def test_windows_never_cross_units(train):
    """Each unit's features must match features computed on that unit alone."""
    out, scaler = make_features(train, SENSORS, WINDOWS)
    for _, rows in train.groupby("unit"):
        alone, _ = make_features(rows, SENSORS, WINDOWS, scaler=scaler)
        pd.testing.assert_frame_equal(out.loc[rows.index], alone.loc[rows.index])


def test_jump_between_units_does_not_leak():
    """A huge offset on one unit must not change its neighbour's rolling features."""
    df = _engines([1, 2])
    out, scaler = make_features(df, SENSORS, WINDOWS)
    shifted = df.copy()
    shifted.loc[shifted.unit == 1, SENSORS] += 1e4
    out_shifted, _ = make_features(shifted, SENSORS, WINDOWS, scaler=scaler)
    unit2 = df.unit == 2
    cols = feature_groups(SENSORS, WINDOWS)["rolling"] + feature_groups(SENSORS, WINDOWS)["slope"]
    pd.testing.assert_frame_equal(out.loc[unit2, cols], out_shifted.loc[unit2, cols])


def test_slope_is_least_squares_over_window(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    unit = out[out.unit == 3].sort_values("cycle")
    x, y = unit.cycle.to_numpy(), unit.s_3_norm.to_numpy()
    for end in (1, 4, 9, 15):
        start = max(0, end - 9)
        expected = np.polyfit(x[start:end + 1], y[start:end + 1], 1)[0]
        assert unit.s_3_slope_10.iloc[end] == pytest.approx(expected)


def test_rolling_mean_and_std_partial_windows(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    unit = out[out.unit == 2].sort_values("cycle")
    y = unit.s_2_norm.to_numpy()
    assert unit.s_2_mean_5.iloc[2] == pytest.approx(y[:3].mean())
    assert unit.s_2_std_5.iloc[2] == pytest.approx(y[:3].std(ddof=1))
    assert unit.s_2_mean_5.iloc[9] == pytest.approx(y[5:10].mean())


def test_scaler_fit_on_train_is_reused_on_validation(train):
    """Validation features use training statistics, so changing validation cannot move them."""
    val = _engines([7, 8], seed=1)
    _, scaler = make_features(train, SENSORS, WINDOWS)
    before = scaler.regime_mean.copy()
    out, returned = make_features(val, SENSORS, WINDOWS, scaler=scaler)
    assert returned is scaler
    pd.testing.assert_frame_equal(scaler.regime_mean, before)
    expected = (val["s_3"] - before.loc[0, "s_3"]) / scaler.regime_std.loc[0, "s_3"]
    np.testing.assert_allclose(out["s_3_norm"], expected)
    _, refit = make_features(val, SENSORS, WINDOWS)
    assert not np.allclose(refit.regime_mean.to_numpy(), before.to_numpy())


def test_health_index_is_bounded_and_falls_with_wear(train):
    out, _ = make_features(train, SENSORS, WINDOWS)
    assert out.health_index.between(0, 1).all()
    assert np.corrcoef(out.cycle, out.health_index)[0, 1] < 0


def test_regimes_found_from_settings():
    """Two operating conditions with offset sensors are split and normalised per regime."""
    df = _engines([1, 2, 3, 4])
    high = df.unit % 2 == 0
    df.loc[high, "setting_3"] = 60.0
    df.loc[high, "s_3"] += 200
    out, _ = make_features(df, SENSORS, WINDOWS, n_regimes=2)
    assert out.groupby("unit")["regime"].nunique().eq(1).all()
    assert out.loc[high, "regime"].nunique() == 1
    assert out.loc[high, "regime"].iloc[0] != out.loc[~high, "regime"].iloc[0]
    assert abs(out["s_3_norm"].mean()) < 1e-9


def test_constant_sensor_is_not_amplified(train):
    """A sensor that is constant up to float noise (s_5, s_16 in FD001) normalises to about 0."""
    df = train.assign(s_16=0.03 + np.tile([0.0, 1e-17], len(train))[:len(train)])
    out, _ = make_features(df, [*SENSORS, "s_16"], WINDOWS)
    assert out["s_16_norm"].abs().max() < 1e-8


def test_mismatched_sensors_rejected(train):
    _, scaler = make_features(train, SENSORS, WINDOWS)
    with pytest.raises(ValueError):
        make_features(train, SENSORS[:2], WINDOWS, scaler=scaler)
