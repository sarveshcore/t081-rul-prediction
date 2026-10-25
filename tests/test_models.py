# ABOUTME: Tests for supported regression model constructors.
# ABOUTME: Checks parameter passthrough, Elastic Net scaling, and invalid names.
from typing import Any, cast

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rul.models import build_model


def test_build_elastic_net_pipeline():
    model = build_model("elastic_net", alpha=0.2, l1_ratio=0.7)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["model"], ElasticNet)
    assert model.named_steps["model"].alpha == 0.2
    assert model.named_steps["model"].l1_ratio == 0.7


def test_build_random_forest_is_seeded_and_accepts_params():
    model = build_model("random_forest", n_estimators=25, max_depth=4)

    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == 42
    assert model.n_estimators == 25
    assert model.max_depth == 4


def test_build_xgboost_is_seeded_and_accepts_params(monkeypatch):
    import sys
    from types import SimpleNamespace

    class FakeXGBRegressor:
        def __init__(self, **params):
            self.params = params

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))
    model = build_model("xgboost", n_estimators=25, max_depth=4)

    assert isinstance(model, FakeXGBRegressor)
    assert model.params == {"n_estimators": 25, "max_depth": 4, "random_state": 42}


def test_model_seed_can_be_overridden():
    model = build_model("random_forest", random_state=7)
    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == 7


def test_unknown_model_fails():
    with pytest.raises(ValueError, match="unknown model"):
        build_model(cast(Any, "svm"))
