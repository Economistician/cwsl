from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from cwsl import ElectricBarometer


def test_electric_barometer_basic_fit_and_predict():
    rng = np.random.RandomState(0)

    # Synthetic regression problem
    n_samples = 200
    n_features = 3

    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1

    # Raw relationship (can be negative)
    y_raw = 5.0 * X[:, 0] + noise

    # Shift so that all "demand" is strictly positive for CWSL
    # This keeps the relative structure but respects the metric domain.
    y = y_raw - y_raw.min() + 1.0  # min(y) > 0

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

    # Basic sanity checks on fitted selector
    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    # results_ should be a DataFrame with one row per model
    assert hasattr(eb, "results_")
    assert isinstance(eb.results_, pd.DataFrame)

    # The index should contain the model names we passed in
    assert set(eb.results_.index) == set(models.keys())

    # Validation metrics for the winning model should be populated
    row = eb.results_.loc[eb.best_name_]

    assert eb.validation_cwsl_ is not None
    assert eb.validation_rmse_ is not None
    assert eb.validation_wmape_ is not None

    assert np.isclose(eb.validation_cwsl_, row["CWSL"])
    assert np.isclose(eb.validation_rmse_, row["RMSE"])
    assert np.isclose(eb.validation_wmape_, row["wMAPE"])

    # Predict on validation set and check shape
    y_pred = eb.predict(X_val)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_val.shape

    # CWSL on the chosen model should be finite and non-negative
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

    # Shift to strictly positive "demand"
    y = y_raw - y_raw.min() + 1.0

    # Simple 80/40 split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
    }

    # Enable refit_on_full at construction time
    eb = ElectricBarometer(
        models=models,
        cu=2.0,
        co=1.0,
        tau=2.0,
        refit_on_full=True,
    )

    # This should run without error and refit the chosen model on full data
    eb.fit(X_train, y_train, X_val, y_val)

    assert eb.best_name_ in models
    assert eb.best_model_ is not None

    # Predictions on the validation set should still be well-formed
    y_pred_val = eb.predict(X_val)
    assert isinstance(y_pred_val, np.ndarray)
    assert y_pred_val.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred_val)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0