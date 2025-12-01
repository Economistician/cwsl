from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from cwsl import CWSLRegressor, cwsl


def _make_positive_regression(
    rng: np.random.RandomState,
    n_samples: int = 200,
    n_features: int = 3,
    coef_first: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple helper: linear-ish signal on the first feature,
    shifted so that all targets are strictly positive.
    """
    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1
    y_raw = coef_first * X[:, 0] + noise
    y = y_raw - y_raw.min() + 1.0
    return X, y


def test_cwsl_regressor_cv_mode_basic():
    rng = np.random.RandomState(0)
    X, y = _make_positive_regression(rng, n_samples=180, n_features=4)

    models = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        random_state=42,
    )

    reg.fit(X, y)

    assert reg.best_name_ in models
    assert reg.best_estimator_ is not None
    assert reg.selector_ is not None
    assert reg.results_ is not None

    # Check validation metrics are finite
    assert np.isfinite(reg.validation_cwsl_)
    assert np.isfinite(reg.validation_rmse_)
    assert np.isfinite(reg.validation_wmape_)

    # Predict on training data
    y_pred = reg.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape

    # Score should be negative CWSL
    score = reg.score(X, y)
    manual_cost = cwsl(y_true=y, y_pred=y_pred, cu=2.0, co=1.0)
    assert np.isclose(score, -manual_cost)


def test_cwsl_regressor_holdout_mode_basic():
    rng = np.random.RandomState(123)
    X, y = _make_positive_regression(rng, n_samples=150, n_features=2)

    models = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    reg = CWSLRegressor(
        models=models,
        cu=3.0,
        co=1.0,
        selection_mode="holdout",
        validation_fraction=0.25,
        refit_on_full=True,
        random_state=7,
    )

    reg.fit(X, y)

    assert reg.best_name_ in models
    assert reg.best_estimator_ is not None
    assert reg.selector_ is not None
    assert reg.results_ is not None

    # Shapes OK on prediction
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape

    # Score defined and finite
    score = reg.score(X, y)
    assert np.isfinite(score)


def test_cwsl_regressor_get_set_params_round_trip():
    rng = np.random.RandomState(999)
    X, y = _make_positive_regression(rng, n_samples=60, n_features=2)

    models = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    reg = CWSLRegressor(
        models=models,
        cu=2.5,
        co=1.0,
        selection_mode="cv",
        cv=4,
        random_state=123,
    )

    params = reg.get_params()
    assert params["cu"] == 2.5
    assert params["cv"] == 4
    assert params["selection_mode"] == "cv"

    # Modify a couple of params and refit
    reg.set_params(cu=3.5, selection_mode="holdout", validation_fraction=0.3)
    assert reg.cu == 3.5
    assert reg.selection_mode == "holdout"
    assert np.isclose(reg.validation_fraction, 0.3)

    reg.fit(X, y)
    assert reg.best_estimator_ is not None