# ABOUTME: Model constructors for remaining useful life regression.
# ABOUTME: Provides consistent Elastic Net, Random Forest, and XGBoost estimators.
from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: Literal["elastic_net", "random_forest", "xgboost"], **params):
    """Build a supported regression model with the given estimator parameters."""
    if name == "elastic_net":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**params)),
            ]
        )
    if name == "random_forest":
        params.setdefault("random_state", 42)
        return RandomForestRegressor(**params)
    if name == "xgboost":
        from xgboost import XGBRegressor

        params.setdefault("random_state", 42)
        return XGBRegressor(**params)
    raise ValueError(f"unknown model {name!r}; choose elastic_net, random_forest, or xgboost")
