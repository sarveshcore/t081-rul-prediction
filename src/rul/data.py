# ABOUTME: Load NASA C-MAPSS FD001 engine trajectories and test RUL labels.
# ABOUTME: Calculate per-cycle RUL targets for run-to-failure training engines.
from pathlib import Path

import pandas as pd

SETTINGS: list[str] = [f"setting_{number}" for number in range(1, 4)]
SENSORS: list[str] = [f"s_{number}" for number in range(1, 22)]
USEFUL_SENSORS: list[str] = SENSORS.copy()  # Refined by the EDA in issue #3.

_COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_fd(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return training and test cycles plus true RUL at each test unit's last cycle."""
    if subset != "FD001":
        raise ValueError("Only FD001 is available; expected subset='FD001'")

    train = pd.read_csv(_RAW_DIR / f"train_{subset}.txt", sep=r"\s+", header=None)
    test = pd.read_csv(_RAW_DIR / f"test_{subset}.txt", sep=r"\s+", header=None)
    test_targets = pd.read_csv(_RAW_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None)

    if train.shape[1] != len(_COLUMNS) or test.shape[1] != len(_COLUMNS):
        raise ValueError(f"{subset} trajectory files must each have {len(_COLUMNS)} columns")
    if test_targets.shape[1] != 1:
        raise ValueError(f"{subset} test RUL file must have one column")

    train.columns = _COLUMNS
    test.columns = _COLUMNS

    units = sorted(test["unit"].unique())
    if len(units) != len(test_targets):
        raise ValueError(f"{subset} has {len(units)} test units but {len(test_targets)} RUL values")

    test_rul = pd.Series(test_targets.iloc[:, 0].to_numpy(), index=pd.Index(units, name="unit"),
                         name="rul")
    return train, test, test_rul


def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add run-to-failure RUL for each cycle, optionally clipped at an upper cap."""
    result = df.copy()
    result["rul"] = result.groupby("unit")["cycle"].transform("max") - result["cycle"]
    if cap is not None:
        result["rul"] = result["rul"].clip(upper=cap)
    return result
