# ABOUTME: Tests for sensor windows: one per cycle, inside engine boundaries, left padded.
# ABOUTME: Also checks scaler reuse, last windows per unit and the CV prepare function.
import numpy as np
import pandas as pd
import pytest

from rul.sequences import last_windows, make_windows, window_prepare

SENSORS = ["s_a", "s_b"]


def toy_frame() -> pd.DataFrame:
    """Unit 1 has cycles 1 to 5, unit 2 cycles 1 to 3; s_a encodes 100 * unit + cycle."""
    rows = [(u, c) for u, n in ((1, 5), (2, 3)) for c in range(1, n + 1)]
    df = pd.DataFrame(rows, columns=["unit", "cycle"])
    df["s_a"] = 100.0 * df["unit"] + df["cycle"]
    df["s_b"] = -df["s_a"]
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def raw(X: np.ndarray, scaler) -> np.ndarray:
    """Undo scaling so tests can read the original sensor values."""
    return scaler.inverse_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)


def test_one_window_per_cycle():
    X, y, unit, _ = make_windows(toy_frame(), SENSORS, window=4)
    assert X.shape == (8, 4, 2)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(y, [4, 3, 2, 1, 0, 2, 1, 0])
    np.testing.assert_array_equal(unit, [1] * 5 + [2] * 3)


def test_windows_never_cross_engines():
    X, _, unit, scaler = make_windows(toy_frame(), SENSORS, window=4)
    s_a = np.rint(raw(X, scaler)[:, :, 0])
    assert (s_a // 100 == unit[:, None]).all()


def test_short_engine_left_padded_with_first_row():
    X, _, _, scaler = make_windows(toy_frame(), SENSORS, window=4)
    s_a = np.rint(raw(X, scaler)[:, :, 0])
    np.testing.assert_array_equal(s_a[5], [201, 201, 201, 201])
    np.testing.assert_array_equal(s_a[7], [201, 201, 202, 203])


def test_window_ends_at_its_own_cycle():
    df = toy_frame()
    X, _, _, scaler = make_windows(df, SENSORS, window=4)
    np.testing.assert_allclose(raw(X, scaler)[:, -1, 0], df["s_a"], rtol=1e-5)


def test_unsorted_input_is_sorted():
    df = toy_frame()
    shuffled = df.sample(frac=1, random_state=42)
    for a, b in zip(make_windows(df, SENSORS, 4)[:3], make_windows(shuffled, SENSORS, 4)[:3]):
        np.testing.assert_array_equal(a, b)


def test_given_scaler_is_not_refit():
    _, _, _, scaler = make_windows(toy_frame(), SENSORS, window=4)
    mean = scaler.mean_.copy()
    shifted = toy_frame().assign(s_a=lambda d: d["s_a"] + 1000)
    _, _, _, returned = make_windows(shifted, SENSORS, window=4, scaler=scaler)
    assert returned is scaler
    np.testing.assert_array_equal(scaler.mean_, mean)


def test_last_windows_one_per_unit_sorted():
    df = toy_frame()
    _, _, _, scaler = make_windows(df, SENSORS, window=4)
    X, units = last_windows(df.sample(frac=1, random_state=42), SENSORS, 4, scaler)
    assert list(units) == [1, 2]
    assert X.shape == (2, 4, 2)
    np.testing.assert_array_equal(np.rint(raw(X, scaler)[:, -1, 0]), [105, 203])
    np.testing.assert_array_equal(np.rint(raw(X, scaler)[1, :, 0]), [201, 201, 202, 203])


def test_last_windows_refuses_to_fit_its_own_scaler():
    with pytest.raises(ValueError, match="training engines"):
        last_windows(toy_frame(), SENSORS, 4)


def test_window_prepare_fits_on_fit_df_only():
    fit_df = toy_frame()
    apply_df = fit_df.assign(s_a=fit_df["s_a"] + 1000, s_b=fit_df["s_b"] - 1000)
    X, y = window_prepare(SENSORS, window=4)(fit_df, apply_df)
    _, _, _, scaler = make_windows(fit_df, SENSORS, window=4)
    X_expected, y_expected, _, _ = make_windows(apply_df, SENSORS, window=4, scaler=scaler)
    np.testing.assert_array_equal(X, X_expected)
    np.testing.assert_array_equal(y, y_expected)
