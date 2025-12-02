from __future__ import annotations

import numpy as np
import pytest

from cwsl import AutoEngine, ElectricBarometer


def _make_positive_regression(
    rng: np.random.RandomState,
    n_samples: int,
    n_features: int,
    coef_first: float = 4.0,
):
    """
    Simple helper to create a synthetic regression problem with
    strictly positive targets (for CWSL validity).
    """
    X = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples) * 0.1
    y_raw = coef_first * X[:, 0] + noise
    y = y_raw - y_raw.min() + 1.0  # ensure strictly positive
    return X, y


def test_auto_engine_build_selector_and_run_holdout():
    rng = np.random.RandomState(0)
    n_samples = 220
    n_features = 4

    X, y = _make_positive_regression(rng, n_samples=n_samples, n_features=n_features)

    # Train/validation split
    X_train, X_val = X[:180], X[180:]
    y_train, y_val = y[:180], y[180:]

    ae = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        cv=3,
        random_state=42,
    )

    eb = ae.build_selector(X_train, y_train)

    # Basic type + config checks
    assert isinstance(eb, ElectricBarometer)
    assert eb.cu == pytest.approx(2.0)
    assert eb.co == pytest.approx(1.0)
    assert eb.selection_mode == "holdout"

    # We should have at least the core tabular engines
    model_names = set(eb.models.keys())
    expected_core = {"dummy_mean", "linear", "ridge", "lasso", "rf", "gbr"}
    assert expected_core.issubset(model_names)

    # Full holdout run should complete and populate metrics
    eb.fit(X_train, y_train, X_val, y_val)

    assert eb.best_name_ in eb.models
    assert eb.best_model_ is not None
    assert eb.results_ is not None

    # predictions + CWSL sanity
    y_pred = eb.predict(X_val)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


def test_auto_engine_cv_selection():
    rng = np.random.RandomState(123)
    n_samples = 180
    n_features = 3

    X, y = _make_positive_regression(rng, n_samples=n_samples, n_features=n_features)

    ae = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        random_state=7,
    )

    eb = ae.build_selector(X, y)

    # CV selection: X/y passed for both train/val slots, val is ignored
    eb.fit(X, y, X, y)

    assert eb.best_name_ in eb.models
    assert eb.best_model_ is not None
    assert eb.results_ is not None

    # Results index should match model dict keys
    model_names = set(eb.models.keys())
    assert set(eb.results_.index) == model_names

    # validation_* should be finite
    assert np.isfinite(eb.validation_cwsl_)
    assert np.isfinite(eb.validation_rmse_)
    assert np.isfinite(eb.validation_wmape_)

    # Predictions sanity
    y_pred = eb.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape

    cwsl_val = eb.cwsl_score(y_true=y, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0


def test_auto_engine_respects_model_family_toggles():
    """
    If we disable all but one family, AutoEngine should build a small,
    predictable model zoo.
    """
    rng = np.random.RandomState(999)
    X, y = _make_positive_regression(rng, n_samples=120, n_features=2)

    ae = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        cv=3,
        random_state=0,
        # Only trees enabled
        use_dummy=False,
        use_linear=False,
        use_regularized_linear=False,
        use_trees=True,
        use_gbm=False,
        use_xgboost=False,
        use_lightgbm=False,
        use_catboost=False,
    )

    eb = ae.build_selector(X, y)

    # With only trees enabled, we expect exactly one model: "rf"
    assert set(eb.models.keys()) == {"rf"}

    # Quick holdout fit to ensure the single-model selector still runs
    X_train, X_val = X[:90], X[90:]
    y_train, y_val = y[:90], y[90:]

    eb.fit(X_train, y_train, X_val, y_val)

    assert eb.best_name_ == "rf"
    assert eb.best_model_ is not None

    y_pred = eb.predict(X_val)
    assert y_pred.shape == y_val.shape

    cwsl_val = eb.cwsl_score(y_true=y_val, y_pred=y_pred)
    assert np.isfinite(cwsl_val)
    assert cwsl_val >= 0.0