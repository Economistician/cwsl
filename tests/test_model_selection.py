import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import select_model_by_cwsl, cwsl


class FixedPredictor:
    """
    Tiny estimator-like class for testing.

    It ignores X/y during fit and always returns a fixed prediction
    vector on predict(X).
    """
    def __init__(self, y_pred_val):
        self.y_pred_val = np.asarray(y_pred_val, dtype=float)

    def fit(self, X, y):
        # No-op fit, just return self
        return self

    def predict(self, X):
        n = len(X)
        if n != len(self.y_pred_val):
            raise ValueError("Length of X does not match stored predictions.")
        return self.y_pred_val


def test_select_model_by_cwsl_prefers_over_lean_model():
    # Validation targets
    y_val = np.array([10, 12, 15, 20, 25, 22, 18, 14, 10], dtype=float)

    # Under-lean model: tends to under-forecast
    y_pred_under = np.array([9, 11, 14, 18, 22, 19, 16, 13, 9], dtype=float)

    # Over-lean model: tends to over-forecast slightly
    y_pred_over = np.array([11, 13, 16, 22, 28, 25, 20, 15, 11], dtype=float)

    # Dummy training data (not actually used by FixedPredictor)
    X_train = np.zeros((5, 1))
    y_train = np.zeros(5)

    # X_val just needs the correct length
    X_val = np.zeros((len(y_val), 1))

    cu = 2.0
    co = 1.0

    models = {
        "under_lean": FixedPredictor(y_pred_under),
        "over_lean": FixedPredictor(y_pred_over),
    }

    best_name, best_model, results = select_model_by_cwsl(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cu=cu,
        co=co,
    )

    # Sanity: results has expected index and columns
    assert set(results.index) == {"under_lean", "over_lean"}
    for col in ["CWSL", "RMSE", "wMAPE"]:
        assert col in results.columns

    # CWSL should penalize under-lean more heavily (R = 2 with cu=2, co=1)
    c_under = results.loc["under_lean", "CWSL"]
    c_over = results.loc["over_lean", "CWSL"]
    assert c_over < c_under

    # And the function should pick the over_lean model
    assert best_name == "over_lean"
    assert isinstance(best_model, FixedPredictor)

    # Double-check CWSL matches a direct call for the chosen model
    direct_cwsl = cwsl(y_true=y_val, y_pred=y_pred_over, cu=cu, co=co)
    assert np.isclose(c_over, direct_cwsl)