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
# Diagnostics
# ─────────────────────────

def nsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    No-Shortfall Level (NSL).

    NSL is the proportion of intervals with no shortfall, optionally weighted.

    A "hit" is defined as y_pred >= y_true for that interval.
    With weights, NSL is the weighted fraction of hits.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, the NSL is
        computed as sum(w_i * hit_i) / sum(w_i). If the total weight is zero,
        a ValueError is raised.

    Returns
    -------
    float
        No-Shortfall Level in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
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
    w = _handle_sample_weight(sample_weight, n)

    # Hit = no shortfall → y_pred >= y_true
    hits = (y_pred_arr >= y_true_arr).astype(float)

    weighted_hits = w * hits
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "NSL is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_hits.sum() / total_weight)


def ud(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Underbuild Depth (UD).

    UD measures the average shortfall depth per interval, optionally weighted.

    For each interval i:
        shortfall_i = max(0, y_true[i] - y_pred[i])

    Unweighted UD:
        UD = mean(shortfall_i)

    Weighted UD:
        UD_w = sum(w_i * shortfall_i) / sum(w_i)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, UD is
        computed as a weighted average. If the total weight is zero, a
        ValueError is raised.

    Returns
    -------
    float
        Underbuild depth (average shortfall per interval).

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
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
    w = _handle_sample_weight(sample_weight, n)

    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)

    weighted_shortfall = w * shortfall
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "UD is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_shortfall.sum() / total_weight)


def wmape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Weighted Mean Absolute Percentage Error (wMAPE).

    Defined here as:

        wMAPE = sum(w_i * |y_true[i] - y_pred[i]|) / sum(w_i * y_true[i])

    This formulation:
    - Allows y_true to be zero for some intervals.
    - Requires total (weighted) demand > 0 to be well-defined.
    - Returns 0.0 if both total (weighted) demand and total (weighted) error are zero.
    - Raises if total (weighted) demand is zero but error is positive.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, wMAPE is
        computed using weighted error and weighted demand. If the total
        weighted demand is zero while error is positive, a ValueError is raised.

    Returns
    -------
    float
        Weighted mean absolute percentage error.

    Raises
    ------
    ValueError
        If inputs are invalid or wMAPE is undefined given the data.
    """
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
    w = _handle_sample_weight(sample_weight, n)

    abs_error = np.abs(y_true_arr - y_pred_arr)

    weighted_error = w * abs_error
    weighted_demand = w * y_true_arr

    total_error = float(weighted_error.sum())
    total_demand = float(weighted_demand.sum())

    if total_demand > 0:
        return total_error / total_demand

    # total_demand == 0
    if total_error == 0:
        # No demand and no error → define wMAPE as 0.0
        return 0.0

    # Error but no demand → undefined under this formulation
    raise ValueError(
        "wMAPE is undefined: total (weighted) demand is zero while total (weighted) "
        "error is positive. Check your data slice or weighting scheme."
    )


def hr_at_tau(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tau: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Hit Rate within Tolerance (HR@τ).

    HR@τ is the proportion of intervals where the absolute error
    is less than or equal to a specified tolerance τ, optionally weighted.

        hit_i = 1 if |y_true[i] - y_pred[i]| <= tau_i
                0 otherwise

    Unweighted HR@τ:
        HR = mean(hit_i)

    Weighted HR@τ:
        HR_w = sum(w_i * hit_i) / sum(w_i)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    tau : float or array-like of shape (n_samples,)
        Absolute error tolerance. Must be non-negative. Can be:
        - scalar: same tolerance for all intervals
        - 1D array: per-interval tolerance

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, HR@τ is
        computed as a weighted average. If the total weight is zero, a
        ValueError is raised.

    Returns
    -------
    float
        Hit rate within tolerance in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
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
    w = _handle_sample_weight(sample_weight, n)

    # Broadcast tau
    tau_arr = _broadcast_param(tau, n, "tau")
    if np.any(tau_arr < 0):
        raise ValueError("tau must be non-negative.")

    abs_error = np.abs(y_true_arr - y_pred_arr)

    hits = (abs_error <= tau_arr).astype(float)

    weighted_hits = w * hits
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "HR@τ is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_hits.sum() / total_weight)


def frs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Forecast Readiness Score (FRS).

    Defined as:

        FRS = NSL - CWSL

    where:
        - NSL is the No-Shortfall Level (fraction of intervals with no shortfall),
        - CWSL is the Cost-Weighted Service Loss, using the same cu/co penalties.

    Higher FRS indicates a forecast that both:
        - avoids shortfalls (high NSL), and
        - avoids costly asymmetric error (low CWSL).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit. Must be strictly positive.
        Must match the cu used for CWSL if you are comparing values.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit. Must be strictly positive.
        Must match the co used for CWSL if you are comparing values.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. Applied consistently to
        both NSL and CWSL.

    Returns
    -------
    float
        Forecast Readiness Score, typically in the range [-inf, 1].
        In practice, values closer to 1 indicate strong readiness.

    Raises
    ------
    ValueError
        If inputs are invalid or CWSL is undefined given the data.
    """
    # We rely on the existing validation in nsl() and cwsl()
    nsl_val = nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    cwsl_val = cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu,
        co=co,
        sample_weight=sample_weight,
    )
    return float(nsl_val - cwsl_val)


# ─────────────────────────
# Baseline Symmetric Metrics
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