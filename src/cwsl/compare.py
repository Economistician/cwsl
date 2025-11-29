from __future__ import annotations

from typing import Iterable, Mapping, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    cwsl,
    nsl,
    ud,
    wmape,
    hr_at_tau,
    frs,
    mae,
    rmse,
    mape,
)

ArrayLike = Union[Iterable[float], np.ndarray]


def compare_forecasts(
    y_true: ArrayLike,
    forecasts: Mapping[str, ArrayLike],
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
    tau: Union[float, ArrayLike] = 2.0,
) -> pd.DataFrame:
    """
    Compare multiple forecast models on the same target series using
    CWSL and related metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand values.

    forecasts : mapping from str to array-like
        Dictionary mapping model names to their forecasted values.
        Each value must be array-like of shape (n_samples,).

        Example:
            {
                "model_a": [9, 15, 7],
                "model_b": [10, 12, 8],
            }

    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval.

    tau : float or array-like, optional (default = 2.0)
        Tolerance parameter for HR@τ.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by model name with columns:

        - CWSL
        - NSL
        - UD
        - wMAPE
        - HR@tau
        - FRS
        - MAE
        - RMSE
        - MAPE

    Raises
    ------
    ValueError
        If forecasts is empty or y_true has invalid shape.
    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional; got shape {y_true_arr.shape}")

    if not forecasts:
        raise ValueError("forecasts mapping is empty; provide at least one model.")

    rows: dict[str, dict[str, float]] = {}

    for model_name, y_pred in forecasts.items():
        # We rely on the metric functions to validate shapes and values.
        metrics_row = {
            "CWSL": cwsl(y_true_arr, y_pred, cu=cu, co=co, sample_weight=sample_weight),
            "NSL": nsl(y_true_arr, y_pred, sample_weight=sample_weight),
            "UD": ud(y_true_arr, y_pred, sample_weight=sample_weight),
            "wMAPE": wmape(y_true_arr, y_pred, sample_weight=sample_weight),
            "HR@tau": hr_at_tau(y_true_arr, y_pred, tau=tau, sample_weight=sample_weight),
            "FRS": frs(y_true_arr, y_pred, cu=cu, co=co, sample_weight=sample_weight),
            "MAE": mae(y_true_arr, y_pred, sample_weight=sample_weight),
            "RMSE": rmse(y_true_arr, y_pred, sample_weight=sample_weight),
            "MAPE": mape(y_true_arr, y_pred, sample_weight=sample_weight),
        }
        rows[model_name] = metrics_row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    return df


def select_model_by_cwsl(
    models: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    cu: float,
    co: float,
    sample_weight_val=None,
) -> Tuple[str, Any, pd.DataFrame]:
    """
    Fit multiple models normally, then select the best one based on CWSL
    evaluated on a validation set.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from model name to an unfitted estimator object that
        implements .fit(X, y) and .predict(X). These can be scikit-learn
        style estimators or any object with that interface.

    X_train, y_train :
        Training data used to fit each model using its native loss
        (typically MSE / RMSE).

    X_val, y_val :
        Validation data used only for evaluation. CWSL and baseline
        metrics are computed on (y_val, model.predict(X_val)).

    cu : float
        Underbuild (shortfall) cost per unit for CWSL.

    co : float
        Overbuild (excess) cost per unit for CWSL.

    sample_weight_val : array-like or None, optional
        Optional sample weights for the validation set, passed into
        the metric functions.

    Returns
    -------
    best_name : str
        Name of the model with the lowest CWSL on the validation set.

    best_model : estimator
        The fitted estimator corresponding to `best_name`.

    results : pandas.DataFrame
        DataFrame indexed by model name with columns:
            - 'CWSL'
            - 'RMSE'
            - 'wMAPE'

    Notes
    -----
    This function does *not* change how models are trained. It simply
    uses CWSL as the selection criterion on a validation set, instead of
    relying on symmetric metrics like RMSE alone.
    """
    y_val_arr = np.asarray(y_val, dtype=float)

    rows = []
    best_name: str | None = None
    best_model: Any | None = None
    best_cwsl = np.inf

    for name, model in models.items():
        # Fit the model normally on the training data
        fitted = model.fit(X_train, y_train)

        # Predict on validation set
        y_pred_val = np.asarray(fitted.predict(X_val), dtype=float)

        # Compute metrics on validation set
        cwsl_val = cwsl(
            y_true=y_val_arr,
            y_pred=y_pred_val,
            cu=cu,
            co=co,
            sample_weight=sample_weight_val,
        )
        rmse_val = rmse(
            y_true=y_val_arr,
            y_pred=y_pred_val,
            sample_weight=sample_weight_val,
        )
        wmape_val = wmape(
            y_true=y_val_arr,
            y_pred=y_pred_val,
            sample_weight=sample_weight_val,
        )

        rows.append(
            {
                "model": name,
                "CWSL": cwsl_val,
                "RMSE": rmse_val,
                "wMAPE": wmape_val,
            }
        )

        if cwsl_val < best_cwsl:
            best_cwsl = cwsl_val
            best_name = name
            best_model = fitted

    results = pd.DataFrame(rows).set_index("model")

    if best_name is None or best_model is None:
        raise ValueError("No models were evaluated. Check the `models` dict.")

    return best_name, best_model, results