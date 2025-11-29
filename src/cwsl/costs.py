from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

from .metrics import _to_1d_array, _handle_sample_weight, _broadcast_param

ArrayLike = Union[Iterable[float], np.ndarray]


def estimate_R_cost_balance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    R_grid: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, ArrayLike] = 1.0,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Estimate a global cost ratio R = cu / co via cost balance.

    For each candidate R in R_grid:

        cu_i = R * co_i

        shortfall_i = max(0, y_true[i] - y_pred[i])
        overbuild_i = max(0, y_pred[i] - y_true[i])

        under_cost(R) = sum(w_i * cu_i * shortfall_i)
        over_cost(R)  = sum(w_i * co_i * overbuild_i)

    We then choose the R that minimizes:

        | under_cost(R) - over_cost(R) |

    Intuition
    ---------
    This "cost balance" method finds the R at which the aggregate
    cost of being short and the aggregate cost of being long are
    as similar as possible for a given forecast and dataset.

    It is a data-driven helper:
    - It does *not* use margin or food cost directly.
    - It depends on the historical error pattern of (y_true, y_pred)
      and the assumed overbuild cost profile co.

    You can use the resulting R* as:
    - a candidate global R for evaluation, or
    - the center of a cost-sensitivity range (e.g., test R in
      {R*/2, R*, 2*R*}).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    R_grid : sequence of float, default=(0.5, 1.0, 2.0, 3.0)
        Candidate cost ratios R to search over. Only strictly positive
        values are considered.

    co : float or array-like of shape (n_samples,), default=1.0
        Overbuild cost per unit. Can be:
        - scalar: same overbuild cost for all intervals
        - 1D array: per-interval overbuild cost

        For each R, underbuild costs are cu_i = R * co_i.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval, used to weight the
        cost aggregation. If None, all intervals get weight 1.0.

    Returns
    -------
    float
        The R in R_grid that minimizes |under_cost(R) - over_cost(R)|.
        If multiple R yield the same minimal gap, the first such value
        in R_grid is returned.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., negative y_true or y_pred),
        R_grid is empty, or contains no positive values.
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

    # Broadcast co and weights
    co_arr = _broadcast_param(co, n, "co")
    w = _handle_sample_weight(sample_weight, n)

    # Precompute shortfall/overbuild
    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)
    overbuild = np.maximum(0.0, y_pred_arr - y_true_arr)

    R_grid_arr = np.asarray(R_grid, dtype=float)
    if R_grid_arr.ndim != 1 or R_grid_arr.size == 0:
        raise ValueError("R_grid must be a non-empty 1D sequence of floats.")

    best_R: float | None = None
    best_gap: float | None = None

    for R in R_grid_arr:
        if R <= 0:
            continue  # ignore non-positive R

        cu_arr = R * co_arr

        under_cost = float(np.sum(w * cu_arr * shortfall))
        over_cost = float(np.sum(w * co_arr * overbuild))

        gap = abs(under_cost - over_cost)

        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_R = float(R)

    if best_R is None:
        raise ValueError(
            "No valid R found in R_grid (ensure it contains at least one positive value)."
        )

    return best_R