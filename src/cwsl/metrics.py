from __future__ import annotations

from typing import Iterable, Union

import numpy as np

ArrayLike = Union[Iterable[float], np.ndarray]


# ─────────────────────────
# Internal helpers
# ─────────────────────────

def _to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a 1D float64 numpy array with basic validation."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")
    return arr


def _broadcast_param(
    param: Union[float, ArrayLike],
    length: int,
    name: str,
) -> np.ndarray:
    """Broadcast a scalar or 1D array-like parameter to match a given length."""
    arr = np.asarray(param, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr))
    elif arr.ndim == 1 and arr.shape[0] == length:
        # already correct shape
        pass
    else:
        raise ValueError(
            f"{name} must be a scalar or 1D array of length {length}; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")
    return arr


def _handle_sample_weight(
    sample_weight: ArrayLike | None,
    length: int,
) -> np.ndarray:
    """Normalize sample_weight to a 1D non-negative array of given length."""
    if sample_weight is None:
        return np.ones(length, dtype=float)

    w = np.asarray(sample_weight, dtype=float)
    if w.ndim == 0:
        w = np.full(length, float(w))
    elif w.ndim == 1 and w.shape[0] == length:
        pass
    else:
        raise ValueError(
            f"sample_weight must be a scalar or 1D array of length {length}; "
            f"got shape {w.shape}"
        )

    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must contain only finite values (no NaN/inf).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")

    return w


# ─────────────────────────
# Core Metric
# ─────────────────────────

def cwsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Cost-Weighted Service Loss (CWSL).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit. Must be strictly positive.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit. Must be strictly positive.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided and the
        total weighted demand is zero while total weighted cost is > 0,
        a ValueError is raised (CWSL undefined in that case).

    Returns
    -------
    float
        Cost-weighted service loss, demand-normalized.

    Raises
    ------
    ValueError
        If inputs are invalid or CWSL is undefined given the data.
    """
    # Convert y_true and y_pred to validated 1D arrays
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if np.any(y_true_arr < 0):
        raise ValueError("y_true must be non-negative (demand cannot be negative).")
    if np.any(y_pred_arr < 0):
        raise ValueError("y_pred must be non-negative (forecast cannot be negative).")

    n = y_true_arr.shape[0]

    # Broadcast cu and co
    cu_arr = _broadcast_param(cu, n, "cu")
    co_arr = _broadcast_param(co, n, "co")

    if np.any(cu_arr <= 0):
        raise ValueError("cu (underbuild cost) must be strictly positive.")
    if np.any(co_arr <= 0):
        raise ValueError("co (overbuild cost) must be strictly positive.")

    # Handle sample_weight
    w = _handle_sample_weight(sample_weight, n)

    # Compute shortfall and overbuild per interval
    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)
    overbuild = np.maximum(0.0, y_pred_arr - y_true_arr)

    # Interval cost
    cost = cu_arr * shortfall + co_arr * overbuild

    # Apply weights
    weighted_cost = w * cost
    weighted_demand = w * y_true_arr

    total_cost = float(weighted_cost.sum())
    total_demand = float(weighted_demand.sum())

    if total_demand > 0:
        return total_cost / total_demand

    # total_demand == 0
    if total_cost == 0:
        # No demand and no cost → define CWSL as 0.0
        return 0.0

    # Cost but no demand → undefined metric under this formulation
    raise ValueError(
        "CWSL is undefined: total (weighted) demand is zero while total (weighted) "
        "cost is positive. Check your data slice or weighting scheme."
    )


# ─────────────────────────
# Diagnostics (stubs for now)
# ─────────────────────────

def nsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    No-Shortfall Level (NSL).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def ud(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Underbuild Depth (UD).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def wmape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Weighted Mean Absolute Percentage Error (wMAPE).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def hr_at_tau(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tau: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Hit Rate within Tolerance (HR@τ).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def frs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Forecast Readiness Score (FRS) = NSL - CWSL.

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


# ─────────────────────────
# Baseline Symmetric Metrics (stubs for now)
# ─────────────────────────

def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Mean Absolute Error (MAE).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Root Mean Squared Error (RMSE).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError


def mape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    NOTE: Implementation to be added.
    """
    raise NotImplementedError