from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from cwsl import AutoEngine, ElectricBarometer


# Optional dependency flags for assertions
HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_CATBOOST = importlib.util.find_spec("catboost") is not None


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
        # speed left as default ("balanced")
    )

    eb = ae.build_selector(X_train, y_train)

    # Basic type + config checks
    assert isinstance(eb, ElectricBarometer)
    assert eb.cu == pytest.approx(2.0)
    assert eb.co == pytest.approx(1.0)
    assert eb.selection_mode == "holdout"

    # We should have at least the core tabular engines
    model_names = set(eb.models.keys())
    expected_core = {"dummy_mean", "linear", "ridge", "rf"}
    # Lasso/GBR may be added depending on preset, but core subset must be present
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
        # default speed = "balanced"
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


def test_auto_engine_speed_presets_change_model_zoo():
    """
    The 'fast', 'balanced', and 'slow' presets should produce nested model zoos:
    fast ⊆ balanced ⊆ slow, and always include a minimal core set.
    """
    rng = np.random.RandomState(999)
    X, y = _make_positive_regression(rng, n_samples=150, n_features=3)

    ae_fast = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        cv=3,
        random_state=0,
        speed="fast",
    )
    ae_balanced = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        cv=3,
        random_state=0,
        speed="balanced",
    )
    ae_slow = AutoEngine(
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        cv=3,
        random_state=0,
        speed="slow",
    )

    eb_fast = ae_fast.build_selector(X, y)
    eb_balanced = ae_balanced.build_selector(X, y)
    eb_slow = ae_slow.build_selector(X, y)

    names_fast = set(eb_fast.models.keys())
    names_balanced = set(eb_balanced.models.keys())
    names_slow = set(eb_slow.models.keys())

    # Core engines should always be present
    core = {"dummy_mean", "linear", "rf"}
    assert core.issubset(names_fast)
    assert core.issubset(names_balanced)
    assert core.issubset(names_slow)

    # Presets should be nested in richness
    assert names_fast.issubset(names_balanced)
    assert names_balanced.issubset(names_slow)

    # Optional dependency-based checks
    if HAS_XGB:
        # If xgboost is installed, it should appear in the richest preset
        assert "xgb" in names_slow

    if HAS_LGBM:
        # LightGBM adapter should also show up in the richest preset when installed
        assert "lgbm" in names_slow

    if HAS_CATBOOST:
        # CatBoost adapter should be part of the richest preset when installed
        assert "catboost" in names_slow

    # Sanity: even in fast mode, a single-model EB still runs end-to-end
    X_train, X_val = X[:100], X[100:]
    y_train, y_val = y[:100], y[100:]

    eb_fast.fit(X_train, y_train, X_val, y_val)

    assert eb_fast.best_name_ in eb_fast.models
    assert eb_fast.best_model_ is not None

    y_pred_fast = eb_fast.predict(X_val)
    assert isinstance(y_pred_fast, np.ndarray)
    assert y_pred_fast.shape == y_val.shape

    cwsl_val_fast = eb_fast.cwsl_score(y_true=y_val, y_pred=y_pred_fast)
    assert np.isfinite(cwsl_val_fast)
    assert cwsl_val_fast >= 0.0