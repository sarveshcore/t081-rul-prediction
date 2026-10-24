# ABOUTME: Feature engineering for C-MAPSS sensor data.
# ABOUTME: Rolling statistics, slopes, regime normalisation, health index.
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from rul.data import SETTINGS

SEED = 42


def norm_col(sensor: str) -> str:
    """Name of the regime normalised copy of a sensor."""
    return f"{sensor}_norm"


def feature_groups(sensors: list[str], windows: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Map each feature family to its column names; every family but raw is added by make_features."""
    return {
        "raw": ["cycle", *sensors],
        "regime": ["regime"],
        "norm": [norm_col(s) for s in sensors],
        "rolling": [f"{s}_{stat}_{w}" for w in windows for s in sensors for stat in ("mean", "std")],
        "slope": [f"{s}_slope_{w}" for w in windows for s in sensors],
        "health": ["health_index"],
    }


def _life_fraction_left(df: pd.DataFrame) -> pd.Series:
    """Health target for training engines: 1 at cycle 1, falling linearly to 0 at failure."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return 1 - (df["cycle"] - 1) / (max_cycle - 1).clip(lower=1)


@dataclass
class FeatureScaler:
    """Training statistics for make_features: regime clusters, per regime moments, HI model."""

    sensors: list[str]
    windows: tuple[int, ...]
    settings_scaler: StandardScaler
    kmeans: KMeans
    regime_mean: pd.DataFrame
    regime_std: pd.DataFrame
    health_model: LinearRegression

    @classmethod
    def fit(cls, df: pd.DataFrame, sensors: list[str], windows: tuple[int, ...],
            n_regimes: int = 1) -> "FeatureScaler":
        """Fit regime clusters, per regime sensor moments and the health index on training rows."""
        settings_scaler = StandardScaler().fit(df[SETTINGS])
        kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=SEED)
        regime = kmeans.fit_predict(settings_scaler.transform(df[SETTINGS]))
        by_regime = df[sensors].groupby(regime)
        regime_mean = by_regime.mean()
        # Constant sensors (std 0 up to float noise) and single row regimes (NaN) are only centred.
        regime_std = by_regime.std().fillna(0)
        regime_std = regime_std.mask(regime_std < 1e-8, 1.0)
        scaler = cls(list(sensors), tuple(windows), settings_scaler, kmeans, regime_mean,
                     regime_std, LinearRegression())
        normed = scaler._normalise(df, regime)
        scaler.health_model.fit(normed, _life_fraction_left(df))
        return scaler

    def regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Assign each row to its nearest training operating regime."""
        return self.kmeans.predict(self.settings_scaler.transform(df[SETTINGS]))

    def _normalise(self, df: pd.DataFrame, regime: np.ndarray) -> pd.DataFrame:
        """Z-score each sensor with the training mean and std of the row's regime."""
        mean = self.regime_mean.loc[regime].to_numpy()
        std = self.regime_std.loc[regime].to_numpy()
        normed = (df[self.sensors].to_numpy(dtype=float) - mean) / std
        return pd.DataFrame(normed, index=df.index, columns=[norm_col(s) for s in self.sensors])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with regime, normalised sensors, rolling stats, slopes and health index."""
        # Work on a (unit, cycle) sorted copy with a clean positional index, then restore order.
        order = np.lexsort((df["cycle"].to_numpy(), df["unit"].to_numpy()))
        data = df.iloc[order].reset_index(drop=True)

        regime = self.regimes(data)
        normed = self._normalise(data, regime)
        health = np.clip(self.health_model.predict(normed), 0, 1)
        rolled = _rolling_features(data["unit"], data["cycle"], normed, self.sensors, self.windows)

        added = pd.concat([pd.Series(regime, name="regime"), normed, rolled,
                           pd.Series(health, name="health_index")], axis=1)
        clash = added.columns.intersection(df.columns)
        if len(clash):
            raise ValueError(f"df already has feature columns, e.g. {list(clash[:3])}")
        added = added.iloc[np.argsort(order)].set_axis(df.index)
        return pd.concat([df, added], axis=1)


def _rolling_features(unit: pd.Series, cycle: pd.Series, normed: pd.DataFrame,
                      sensors: list[str], windows: tuple[int, ...]) -> pd.DataFrame:
    """Rolling mean, std and OLS slope per sensor, each window kept inside one unit."""
    y = normed.to_numpy()
    x = cycle.to_numpy(dtype=float)[:, None]
    # Rolling sums of 1, x, x^2, y and x*y give the least squares slope in closed form.
    parts = np.hstack([np.ones_like(x), x, x**2, y, x * y])
    columns = {}
    for w in windows:
        grouped = pd.DataFrame(parts).groupby(unit.to_numpy(), sort=False)
        # Early cycles use the min_periods=1 partial window: the cycles seen so far.
        sums = grouped.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        sums = sums.sort_index().to_numpy()
        n, sx, sxx = sums[:, [0]], sums[:, [1]], sums[:, [2]]
        sy, sxy = sums[:, 3:3 + len(sensors)], sums[:, 3 + len(sensors):]
        denom = n * sxx - sx**2
        # A one-cycle window has no trend: denom is 0 there and the slope is defined as 0.
        safe = np.where(denom > 0, denom, 1)
        slope = np.where(denom > 0, (n * sxy - sx * sy) / safe, 0.0)
        stats = normed.groupby(unit.to_numpy(), sort=False).rolling(w, min_periods=1)
        mean = stats.mean().reset_index(level=0, drop=True).sort_index().to_numpy()
        # Sample std is undefined for one cycle; a single reading has zero spread.
        std = stats.std().reset_index(level=0, drop=True).sort_index().fillna(0).to_numpy()
        for i, s in enumerate(sensors):
            columns[f"{s}_mean_{w}"] = mean[:, i]
            columns[f"{s}_std_{w}"] = std[:, i]
        for i, s in enumerate(sensors):
            columns[f"{s}_slope_{w}"] = slope[:, i]
    return pd.DataFrame(columns, index=normed.index)


def make_features(df: pd.DataFrame, sensors: list[str], windows: tuple[int, ...] = (5, 10, 20),
                  scaler: object | None = None, n_regimes: int | None = None
                  ) -> tuple[pd.DataFrame, object]:
    """Same rows as df plus features; rolling stats, slopes and HI use regime normalised sensors."""
    # No scaler: fit one on df (training engines only). Given a scaler: apply it, never refit.
    if scaler is None:
        scaler = FeatureScaler.fit(df, sensors, windows, n_regimes=n_regimes or 1)
    elif list(sensors) != scaler.sensors or tuple(windows) != scaler.windows:
        raise ValueError("sensors and windows must match the ones the scaler was fitted with")
    elif n_regimes is not None and n_regimes != scaler.kmeans.n_clusters:
        raise ValueError(f"n_regimes={n_regimes} but the scaler was fitted with "
                         f"{scaler.kmeans.n_clusters} regimes")
    return scaler.transform(df), scaler
