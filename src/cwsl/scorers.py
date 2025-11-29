# pyright: reportMissingModuleSource=false

from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import cwsl


def cwsl_loss(
    y_true,
    y_pred,
    *,
    cu: float,
    co: float,
    sample_weight=None,
) -> float:
    """
    Raw CWSL loss function, suitable for use with sklearn.make_scorer.

    This returns a *positive* cost (loss). When wrapped with
    sklearn.metrics.make_scorer(greater_is_better=False), the resulting
    scorer will return the *negative* CWSL so that higher scores are
    better, as per sklearn conventions.
    """
    return cwsl(
        y_true=np.asarray(y_true, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
        cu=cu,
        co=co,
        sample_weight=sample_weight,
    )


def cwsl_scorer(
    cu: float,
    co: float,
) -> Callable:
    """
    Build a scikit-learn compatible scorer based on CWSL.

    The returned object can be used anywhere sklearn expects a "scorer",
    such as GridSearchCV, RandomizedSearchCV, cross_val_score, etc.

    Notes
    -----
    - The underlying CWSL is a *loss* (lower is better).
    - The scorer returned by this function obeys sklearn conventions:
        * it returns the *negative* CWSL (so higher is better),
        * and can be maximized by sklearn search utilities.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from cwsl import cwsl_scorer
    >>>
    >>> scorer = cwsl_scorer(cu=2.0, co=1.0)
    >>> model = RandomForestRegressor(random_state=0)
    >>> grid = GridSearchCV(
    ...     estimator=model,
    ...     param_grid={"n_estimators": [50, 100]},
    ...     scoring=scorer,
    ... )
    """
    try:
        from sklearn.metrics import make_scorer
    except ImportError as e:
        raise ImportError(
            "cwsl_scorer requires scikit-learn to be installed. "
            "Install it with `pip install scikit-learn`."
        ) from e

    def _loss(y_true, y_pred, sample_weight=None):
        return cwsl_loss(
            y_true=y_true,
            y_pred=y_pred,
            cu=cu,
            co=co,
            sample_weight=sample_weight,
        )

    # greater_is_better=False → sklearn will internally negate this loss,
    # so the scorer returns -CWSL and higher scores are better.
    return make_scorer(
        _loss,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )