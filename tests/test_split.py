# ABOUTME: Tests for engine level splitting in rul.split.
# ABOUTME: Guards against leakage: no engine may appear on both sides of a split.
import numpy as np
import pandas as pd
import pytest

from rul.split import engine_folds, engine_split, last_cycle

UNITS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


@pytest.fixture
def df():
    """Ten engines with uneven ids and lifetimes, rows shuffled."""
    rows = [
        {"unit": u, "cycle": c, "s_2": float(c)}
        for u in UNITS
        for c in range(1, u % 5 + 4)
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def test_split_has_no_shared_units(df):
    train, val = engine_split(df)
    assert set(train.unit) & set(val.unit) == set()
    assert set(train.unit) | set(val.unit) == set(UNITS)


def test_split_size(df):
    _, val = engine_split(df, val_frac=0.2)
    assert val.unit.nunique() == 2


def test_split_keeps_at_least_one_val_unit(df):
    _, val = engine_split(df, val_frac=0.01)
    assert val.unit.nunique() == 1


def test_split_keeps_all_rows(df):
    train, val = engine_split(df)
    assert len(train) + len(val) == len(df)


def test_split_is_seeded(df):
    _, a = engine_split(df, seed=42)
    _, b = engine_split(df, seed=42)
    _, c = engine_split(df, seed=0)
    assert set(a.unit) == set(b.unit)
    assert set(a.unit) != set(c.unit)


def test_folds_cover_each_unit_once(df):
    seen = []
    for train_idx, val_idx in engine_folds(df, n_splits=5):
        train_units = set(df.iloc[train_idx].unit)
        val_units = set(df.iloc[val_idx].unit)
        assert train_units & val_units == set()
        seen.extend(val_units)
    assert sorted(seen) == UNITS


def test_last_cycle(df):
    last = last_cycle(df)
    assert list(last.unit) == UNITS
    expected = df.groupby("unit")["cycle"].max().to_numpy()
    np.testing.assert_array_equal(last.cycle.to_numpy(), expected)
    assert list(last.index) == list(range(len(UNITS)))
