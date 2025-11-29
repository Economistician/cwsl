from __future__ import annotations

from typing import Optional, Union, Dict, Sequence, List

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


def evaluate_hierarchy_df(
    df: pd.DataFrame,
    levels: Dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu,
    co,
    tau: float | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate CWSL and related diagnostics at multiple grouping levels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the actual and forecast columns,
        plus any grouping columns referenced in `levels`.

    levels : dict[str, Sequence[str]]
        Mapping of level name -> list/tuple of column names to group by.

        Examples
        --------
        levels = {
            "overall": [],
            "by_store": ["store_id"],
            "by_item": ["item_id"],
            "by_store_item": ["store_id", "item_id"],
        }

        An empty list means "treat the entire DataFrame as one group".

    actual_col : str
        Column name for actual demand.

    forecast_col : str
        Column name for forecasted demand.

    cu : float or array-like
        Underbuild cost parameter passed to `cwsl`.

    co : float or array-like
        Overbuild cost parameter passed to `cwsl`.

    tau : float, optional
        Tolerance passed to `hr_at_tau`. If None, HR@τ is omitted.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping level name -> DataFrame of metrics for that level.

        Each DataFrame includes:
            - any grouping columns for that level
            - n_intervals
            - total_demand
            - cwsl
            - nsl
            - ud
            - wmape
            - hr_at_tau (if tau is not None)
            - frs
    """
    results: Dict[str, pd.DataFrame] = {}

    # Ensure required columns exist up front
    required_cols = {actual_col, forecast_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    for level_name, group_cols in levels.items():
        group_cols = list(group_cols)  # normalize

        if len(group_cols) == 0:
            # Single overall group
            y_true = df[actual_col].to_numpy(dtype=float)
            y_pred = df[forecast_col].to_numpy(dtype=float)

            metrics_row = {
                "n_intervals": len(df),
                "total_demand": float(df[actual_col].sum()),
                "cwsl": cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co),
                "nsl": nsl(y_true=y_true, y_pred=y_pred),
                "ud": ud(y_true=y_true, y_pred=y_pred),
                "wmape": wmape(y_true=y_true, y_pred=y_pred),
            }
            if tau is not None:
                metrics_row["hr_at_tau"] = hr_at_tau(
                    y_true=y_true,
                    y_pred=y_pred,
                    tau=tau,
                )
            metrics_row["frs"] = frs(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu,
                co=co,
            )

            overall_df = pd.DataFrame([metrics_row])
            results[level_name] = overall_df

        else:
            # Grouped evaluation
            group_rows: List[dict] = []

            grouped = df.groupby(group_cols, dropna=False, sort=False)
            for keys, df_g in grouped:
                # keys is a scalar or tuple of scalars depending on number of group_cols
                if not isinstance(keys, tuple):
                    keys = (keys,)

                y_true = df_g[actual_col].to_numpy(dtype=float)
                y_pred = df_g[forecast_col].to_numpy(dtype=float)

                row = {
                    "n_intervals": len(df_g),
                    "total_demand": float(df_g[actual_col].sum()),
                    "cwsl": cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co),
                    "nsl": nsl(y_true=y_true, y_pred=y_pred),
                    "ud": ud(y_true=y_true, y_pred=y_pred),
                    "wmape": wmape(y_true=y_true, y_pred=y_pred),
                }
                if tau is not None:
                    row["hr_at_tau"] = hr_at_tau(
                        y_true=y_true,
                        y_pred=y_pred,
                        tau=tau,
                    )
                row["frs"] = frs(
                    y_true=y_true,
                    y_pred=y_pred,
                    cu=cu,
                    co=co,
                )

                # Attach grouping keys
                for col, value in zip(group_cols, keys):
                    row[col] = value

                group_rows.append(row)

            level_df = pd.DataFrame(group_rows)
            # Put group columns first
            results[level_name] = level_df[
                list(group_cols)
                + [c for c in level_df.columns if c not in group_cols]
            ]

    return results