from __future__ import annotations

from typing import Optional, Union

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


def compute_cwsl_df(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    cu: Union[float, str],
    co: Union[float, str],
    sample_weight_col: Optional[str] = None,
) -> float:
    """
    Compute CWSL from a pandas DataFrame.

    This is a convenience wrapper around the core `cwsl(...)` function.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the actual and forecast columns.
    y_true_col : str
        Name of the column in `df` containing actual demand.
    y_pred_col : str
        Name of the column in `df` containing forecasted demand.
    cu : float or str
        Either:
        - A scalar cost per unit of underbuild (applied to all rows), OR
        - The name of a column in `df` containing per-row underbuild costs.
    co : float or str
        Either:
        - A scalar cost per unit of overbuild (applied to all rows), OR
        - The name of a column in `df` containing per-row overbuild costs.
    sample_weight_col : str, optional
        If provided, the name of a column in `df` containing non-negative
        sample weights. If None, all rows are weighted equally.

    Returns
    -------
    float
        Cost-weighted service loss, demand-normalized.
    """
    # Extract core series as numpy arrays
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    # Handle cu: scalar vs column name
    if isinstance(cu, str):
        cu_value = df[cu].to_numpy(dtype=float)
    else:
        cu_value = cu

    # Handle co: scalar vs column name
    if isinstance(co, str):
        co_value = df[co].to_numpy(dtype=float)
    else:
        co_value = co

    # Handle optional sample_weight column
    if sample_weight_col is not None:
        sample_weight = df[sample_weight_col].to_numpy(dtype=float)
    else:
        sample_weight = None

    # Delegate to the core implementation
    return cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu_value,
        co=co_value,
        sample_weight=sample_weight,
    )


def evaluate_groups_df(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    cu: float = 2.0,
    co: float = 1.0,
    tau: float = 2.0,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute all core CWSL metrics per group from a pandas DataFrame.

    For each group defined by `group_cols`, this helper computes:
        - CWSL
        - NSL
        - UD
        - wMAPE
        - HR@tau
        - FRS
        - MAE
        - RMSE
        - MAPE

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing actuals, forecasts, and grouping columns.

    group_cols : list of str
        Column names to group by (e.g. ["store_id", "item_id"]).

    actual_col : str, default "actual_qty"
        Name of the column containing actual demand.

    forecast_col : str, default "forecast_qty"
        Name of the column containing forecasted demand.

    cu : float, default 2.0
        Underbuild (shortfall) cost per unit, applied uniformly
        across all rows/groups.

    co : float, default 1.0
        Overbuild (excess) cost per unit, applied uniformly
        across all rows/groups.

    tau : float, default 2.0
        Tolerance parameter for HR@tau (absolute units).

    sample_weight_col : str or None, default None
        Optional column name containing non-negative sample weights
        per row. If provided, all metrics that accept `sample_weight`
        will use this column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per group and columns:
        group_cols + ["CWSL", "NSL", "UD", "wMAPE", "HR@tau",
                      "FRS", "MAE", "RMSE", "MAPE"].

        If a metric is undefined for a particular group (e.g. due to
        invalid data), the corresponding value is NaN rather than
        raising an error for the entire evaluation.
    """
    # Basic column validation
    missing = [c for c in group_cols + [actual_col, forecast_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col '{sample_weight_col}' not found in df")

    results: list[dict] = []

    # Helper to safely compute a metric per group (return NaN on ValueError)
    def _safe_metric(fn) -> float:
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    grouped = df.groupby(group_cols, sort=False)

    for key, g in grouped:
        # Normalize group key into a tuple
        if not isinstance(key, tuple):
            key = (key,)

        row: dict = {col: val for col, val in zip(group_cols, key)}

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            sample_weight = g[sample_weight_col].to_numpy(dtype=float)
        else:
            sample_weight = None

        # Core + diagnostics
        row["CWSL"] = _safe_metric(
            lambda: cwsl(y_true, y_pred, cu=cu, co=co, sample_weight=sample_weight)
        )
        row["NSL"] = _safe_metric(lambda: nsl(y_true, y_pred, sample_weight=sample_weight))
        row["UD"] = _safe_metric(lambda: ud(y_true, y_pred, sample_weight=sample_weight))
        row["wMAPE"] = _safe_metric(
            lambda: wmape(y_true, y_pred, sample_weight=sample_weight)
        )
        row["HR@tau"] = _safe_metric(
            lambda: hr_at_tau(
                y_true,
                y_pred,
                tau=tau,
                sample_weight=sample_weight,
            )
        )
        row["FRS"] = _safe_metric(
            lambda: frs(y_true, y_pred, cu=cu, co=co, sample_weight=sample_weight)
        )

        # Baseline symmetric metrics
        row["MAE"] = _safe_metric(lambda: mae(y_true, y_pred, sample_weight=sample_weight))
        row["RMSE"] = _safe_metric(
            lambda: rmse(y_true, y_pred, sample_weight=sample_weight)
        )
        row["MAPE"] = _safe_metric(
            lambda: mape(y_true, y_pred, sample_weight=sample_weight)
        )

        results.append(row)

    return pd.DataFrame(results)