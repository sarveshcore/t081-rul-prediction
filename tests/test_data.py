# ABOUTME: Tests for loading C-MAPSS FD001 and calculating training RUL.
# ABOUTME: Checks the public data contract and guards against target leakage.
import pandas as pd
import pytest

from rul import data
from rul.data import add_rul, load_fd


def test_load_fd001_has_expected_engines_and_columns():
    train, test, test_rul = load_fd()
    expected = ["unit", "cycle", *(f"setting_{n}" for n in range(1, 4)),
                *(f"s_{n}" for n in range(1, 22))]
    assert list(train.columns) == expected
    assert list(test.columns) == expected
    assert train.shape == (20631, 26)
    assert test.shape == (13096, 26)
    assert train.unit.nunique() == test.unit.nunique() == len(test_rul) == 100
    assert pd.api.types.is_integer_dtype(train.unit)
    assert pd.api.types.is_integer_dtype(test.cycle)


def test_test_rul_maps_archive_rows_to_unit_ids():
    _, test, test_rul = load_fd("FD001")
    assert list(test_rul.index) == sorted(test.unit.unique())
    assert test_rul.loc[1] == 112
    assert test_rul.loc[2] == 98
    assert test_rul.loc[100] == 20


def test_add_rul_uses_each_engines_last_cycle_without_changing_input():
    original = pd.DataFrame({"unit": [7, 3, 7, 3, 7], "cycle": [2, 3, 1, 1, 3]},
                            index=[9, 8, 7, 6, 5])
    result = add_rul(original)
    assert result.rul.tolist() == [1, 0, 2, 2, 0]
    assert result.index.tolist() == original.index.tolist()
    assert "rul" not in original.columns


def test_add_rul_caps_only_values_above_cap():
    df = pd.DataFrame({"unit": [1, 1, 1, 1], "cycle": [1, 2, 3, 4]})
    assert add_rul(df, cap=2).rul.tolist() == [2, 2, 1, 0]
    assert add_rul(df).rul.tolist() == [3, 2, 1, 0]


def test_training_engines_reach_zero_and_decrease_each_cycle():
    train, _, _ = load_fd()
    labeled = add_rul(train)
    assert (labeled.groupby("unit").tail(1).rul == 0).all()
    assert (labeled.groupby("unit").rul.diff().dropna() == -1).all()


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="FD001"):
        load_fd("FD999")


@pytest.mark.parametrize("extra_field", [False, True])
def test_load_fd_rejects_wrong_trajectory_width(tmp_path, monkeypatch, extra_field):
    valid_row = "1 1 " + "0 " * 24
    bad_row = valid_row + "0 " if extra_field else "1 1 " + "0 " * 23
    (tmp_path / "train_FD001.txt").write_text(bad_row + "\n")
    (tmp_path / "test_FD001.txt").write_text(valid_row + "\n")
    (tmp_path / "RUL_FD001.txt").write_text("5\n")
    monkeypatch.setattr(data, "_RAW_DIR", tmp_path)

    with pytest.raises(ValueError, match="26 columns"):
        load_fd()
