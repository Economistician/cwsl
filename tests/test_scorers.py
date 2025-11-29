import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import cwsl, cwsl_loss, cwsl_scorer


class FixedPredictor:
    """
    Tiny estimator-like class for testing sklearn scorers.

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


def test_cwsl_loss_matches_core_metric():
    y_true = np.array([10, 12, 15], dtype=float)
    y_pred = np.array([9, 13, 14], dtype=float)
    cu = 2.0
    co = 1.0

    direct = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
    loss = cwsl_loss(y_true=y_true, y_pred=y_pred, cu=cu, co=co)

    assert np.isclose(direct, loss)


def test_cwsl_scorer_returns_negative_cwsl():
    try:
        from sklearn.metrics import get_scorer
    except ImportError:
        # If sklearn is not installed, skip this test.
        # (In your real setup, sklearn should be installed as a dependency.)
        return

    y_true = np.array([10, 12, 15, 20], dtype=float)

    # Two candidate predictors: one under-lean, one over-lean
    y_pred_under = np.array([9, 11, 14, 18], dtype=float)
    y_pred_over = np.array([11, 13, 16, 22], dtype=float)

    X = np.zeros((len(y_true), 1))  # dummy features

    cu = 2.0
    co = 1.0

    scorer = cwsl_scorer(cu=cu, co=co)

    est_under = FixedPredictor(y_pred_under)
    est_over = FixedPredictor(y_pred_over)

    # scorer(est, X, y) → higher is better
    score_under = scorer(est_under, X, y_true)
    score_over = scorer(est_over, X, y_true)

    # Compute raw CWSL for sanity
    cwsl_under = cwsl(y_true=y_true, y_pred=y_pred_under, cu=cu, co=co)
    cwsl_over = cwsl(y_true=y_true, y_pred=y_pred_over, cu=cu, co=co)

    # Scorer should return the *negative* CWSL
    assert np.isclose(score_under, -cwsl_under)
    assert np.isclose(score_over, -cwsl_over)

    # And, since lower CWSL is better, the over-lean model should have
    # a higher scorer value (less negative).
    assert cwsl_over < cwsl_under
    assert score_over > score_under