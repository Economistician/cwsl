from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional XGBoost support
try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False

from cwsl import ElectricBarometer


def _make_positive_regression(
    rng: np.random.RandomState,
    n_samples: int,
    n_features: int,
    coef_first: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper: create a synthetic regression problem with strictly positive targets.

    We generate a simple linear-ish signal on the first feature and then shift
    everything upwards so that y > 0 for all samples (required for CWSL).
    """
    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1
    y_raw = coef_first * X[:, 0] + noise
    y = y_raw - y_raw.min() + 1.0  # ensure strictly positive "demand"
    return X, y


def test_electric_barometer_basic_fit_and_predict():
    rng = np.random.RandomState(0)

    n_samples = 200
    n_features = 3

    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1

    # Raw relationship (can be negative)
    y_raw = 5.0 * X[:, 0] + noise

    # Shift to strictly positive "demand" for CWSL validity
    y = y_raw - y_raw.min() + 1.0

    # Train/validation split
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    eb = ElectricBarometer(models=models, cu=2.0, co=1.0, tau=2.0)

    # Fit and select best model
    eb.fit(X_train, y_train, X_val, y_val)

    # Basic sanity checks
    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    assert hasattr(eb, "results_")
    assert isinstance(eb.results_, pd.DataFrame)
    assert set(eb.results_.index) == set(models.keys())

    # Validation metrics for the winning model should be populated
    row = eb.results_.loc[eb.best_name_]

    assert eb.validation_cwsl_ is not None
    assert eb.validation_rmse_ is not None
    assert eb.validation_wmape_ is not None

    assert np.isclose(eb.validation_cwsl_, row["CWSL"])
    assert np.isclose(eb.validation_rmse_, row["RMSE"])
    assert np.isclose(eb.validation_wmape_, row["wMAPE"])

    # Predictions from the chosen model
    y_pred = eb.predict(X_val)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


def test_electric_barometer_refit_on_full_runs():
    rng = np.random.RandomState(42)

    n_samples = 120
    n_features = 2

    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1
    y_raw = 3.0 * X[:, 0] - 2.0 * X[:, 1] + noise
    y = y_raw - y_raw.min() + 1.0  # strictly positive

    # Simple 80/40 split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    eb = ElectricBarometer(
        models=models,
        cu=2.0,
        co=1.0,
        tau=2.0,
        refit_on_full=True,
    )

    # Should run without error and refit winning model on full (train ∪ val)
    eb.fit(X_train, y_train, X_val, y_val)

    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    y_pred_val = eb.predict(X_val)
    assert isinstance(y_pred_val, np.ndarray)
    assert y_pred_val.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred_val)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


def test_electric_barometer_cv_selection():
    rng = np.random.RandomState(123)

    n_samples = 150
    n_features = 3

    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1

    y_raw = 4.0 * X[:, 0] + noise
    y = y_raw - y_raw.min() + 1.0

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    eb = ElectricBarometer(
        models=models,
        cu=2.0,
        co=1.0,
        tau=2.0,
        selection_mode="cv",
        cv=3,
        random_state=42,
    )

    # For CV mode we can pass X, y for both train/val slots; val is ignored
    eb.fit(X, y, X, y)

    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    assert hasattr(eb, "results_")
    assert isinstance(eb.results_, pd.DataFrame)
    assert set(eb.results_.index) == set(models.keys())

    for col in ["CWSL", "RMSE", "wMAPE"]:
        assert col in eb.results_.columns

    # validation_* are mean CV scores; they just need to be finite
    assert np.isfinite(eb.validation_cwsl_)
    assert np.isfinite(eb.validation_rmse_)
    assert np.isfinite(eb.validation_wmape_)

    # Predictions from the refit model should look correct
    y_pred = eb.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape

    cwsl_val = eb.cwsl_score(y_true=y, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


def test_electric_barometer_tree_ensemble_engines():
    """EB should work cleanly with non-linear tree/ensemble regressors."""
    rng = np.random.RandomState(7)
    n_samples = 200
    n_features = 5

    X, y = _make_positive_regression(rng, n_samples=n_samples, n_features=n_features)

    # Train/validation split
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    models = {
        "rf": RandomForestRegressor(
            n_estimators=50,
            random_state=0,
        ),
        "gbr": GradientBoostingRegressor(
            random_state=0,
        ),
    }

    eb = ElectricBarometer(models=models, cu=2.0, co=1.0, tau=2.0)

    eb.fit(X_train, y_train, X_val, y_val)

    # Winner must be one of the tree models
    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    # results_ should contain rows for both models
    assert isinstance(eb.results_, pd.DataFrame)
    assert set(eb.results_.index) == set(models.keys())

    # Validation metrics should be finite and sensible
    assert eb.validation_cwsl_ is not None
    assert np.isfinite(eb.validation_cwsl_)
    assert eb.validation_cwsl_ >= 0.0

    assert eb.validation_rmse_ is not None
    assert np.isfinite(eb.validation_rmse_)
    assert eb.validation_rmse_ >= 0.0

    assert eb.validation_wmape_ is not None
    assert np.isfinite(eb.validation_wmape_)
    assert eb.validation_wmape_ >= 0.0

    # Predict on validation set and check shape
    y_pred = eb.predict(X_val)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
def test_electric_barometer_xgboost_engine():
    """EB should work with XGBRegressor via its sklearn API when available."""
    rng = np.random.RandomState(21)
    n_samples = 250
    n_features = 4

    X, y = _make_positive_regression(rng, n_samples=n_samples, n_features=n_features)

    # Train/validation split
    X_train, X_val = X[:200], X[200:]
    y_train, y_val = y[:200], y[200:]

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "xgb": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
        ),
    }

    eb = ElectricBarometer(
        models=models,
        cu=2.0,
        co=1.0,
        tau=2.0,
        selection_mode="holdout",
        refit_on_full=False,
    )

    eb.fit(X_train, y_train, X_val, y_val)

    # We don't require XGB to win, but the pipeline must run cleanly
    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    assert isinstance(eb.results_, pd.DataFrame)
    assert set(eb.results_.index) == set(models.keys())

    assert eb.validation_cwsl_ is not None
    assert np.isfinite(eb.validation_cwsl_)
    assert eb.validation_cwsl_ >= 0.0

    # Predict and score
    y_pred = eb.predict(X_val)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0