from __future__ import annotations

from typing import Iterable, Mapping, Union

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