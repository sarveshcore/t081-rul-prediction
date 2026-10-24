# ABOUTME: Load NASA C-MAPSS FD001 engine trajectories and test RUL labels.
# ABOUTME: Calculate per-cycle RUL targets for run-to-failure training engines.
from pathlib import Path

import pandas as pd

SETTINGS: list[str] = [f"setting_{number}" for number in range(1, 4)]
SENSORS: list[str] = [f"s_{number}" for number in range(1, 22)]
# Training-only EDA drops six constants and the near-constant binary s_6.
# See notebooks/01_eda.ipynb and results/eda_sensor_summary.csv for evidence.
USEFUL_SENSORS: list[str] = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11",
    "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21",
]

_COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _read_trajectory(path: Path) -> pd.DataFrame:
    """Read one raw trajectory file and validate its column count before naming columns."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(_COLUMNS):
        raise ValueError(f"{path.name} must have {len(_COLUMNS)} columns")
    frame.columns = _COLUMNS
    return frame


def load_train_fd(subset: str = "FD001") -> pd.DataFrame:
    """Load only training trajectories, without opening test observations or test labels."""
    if subset != "FD001":
        raise ValueError("Only FD001 is available; expected subset='FD001'")
    return _read_trajectory(_RAW_DIR / f"train_{subset}.txt")


def load_fd(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return training and test cycles plus true RUL at each test unit's last cycle."""
    train = load_train_fd(subset)
    test = _read_trajectory(_RAW_DIR / f"test_{subset}.txt")
    test_targets = pd.read_csv(_RAW_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None)

    if test_targets.shape[1] != 1:
        raise ValueError(f"{subset} test RUL file must have one column")

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
