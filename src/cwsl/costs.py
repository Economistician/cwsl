from __future__ import annotations

from typing import Iterable, Sequence, Union, Optional

import pandas as pd
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


def estimate_entity_R_from_balance(
    df: pd.DataFrame,
    entity_col: str,
    y_true_col: str,
    y_pred_col: str,
    ratios: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float = 1.0,
    sample_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate an entity-level cost ratio R_e = cu_e / co for each entity,
    using a simple "balance" method over a grid of candidate ratios.

    For each entity:
        1. Take all historical intervals for that entity.
        2. For each candidate R in `ratios`:
            - set cu = R * co
            - compute total underbuild cost and overbuild cost:
                under_cost(R) = sum(w_i * cu * shortfall_i)
                over_cost(R)  = sum(w_i * co * overbuild_i)
            - measure the imbalance:
                diff(R) = |under_cost(R) - over_cost(R)|
        3. Choose the R that MINIMIZES diff(R).
        4. Report R_e, cu_e = R_e * co, and the under/over costs at that R_e.

    This does NOT claim that past forecasts were unbiased; it simply
    finds the R where the *observed* underbuild and overbuild costs are
    closest in magnitude under your cost grid. Entities that are strongly
    skewed (mostly under- or over-forecasted) will tend to "hug" the
    edges of the `ratios` range.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing an entity identifier, actuals, and forecasts.

    entity_col : str
        Column in `df` identifying the entity (e.g., "item", "sku",
        "product", "line", "location").

    y_true_col : str
        Column containing actual demand.

    y_pred_col : str
        Column containing forecasted demand.

    ratios : sequence of float, default (0.5, 1.0, 2.0, 3.0)
        Candidate R values to search over for each entity. These should be
        positive. A typical starting grid is something like
        (0.5, 1.0, 2.0, 3.0).

    co : float, default 1.0
        Overbuild (excess) cost per unit, assumed scalar and common
        across entities for this estimation method.

    sample_weight_col : str or None, default None
        Optional column of non-negative sample weights per row.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns:

            - entity_col          (entity identifier)
            - R                   (chosen cost ratio)
            - cu                  (shortfall cost = R * co)
            - co                  (overbuild cost, scalar input)
            - under_cost          (total underbuild cost at chosen R)
            - over_cost           (total overbuild cost at chosen R)
            - diff                (|under_cost - over_cost| at chosen R)

    Notes
    -----
    - This is an "advanced" helper for entity-level tuning. You can still
      use a global R for most use cases, and reserve entity-level R_e for
      high-impact entities or mature users.
    """
    required = {entity_col, y_true_col, y_pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col '{sample_weight_col}' not found in df")

    ratios_arr = np.asarray(list(ratios), dtype=float)
    if ratios_arr.ndim != 1 or np.any(ratios_arr <= 0):
        raise ValueError("ratios must be a 1D sequence of positive floats.")

    if co <= 0:
        raise ValueError("co must be strictly positive.")

    results: list[dict] = []

    grouped = df.groupby(entity_col, sort=False)

    for entity_id, g in grouped:
        y_true = g[y_true_col].to_numpy(dtype=float)
        y_pred = g[y_pred_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            w = g[sample_weight_col].to_numpy(dtype=float)
        else:
            w = np.ones_like(y_true, dtype=float)

        # Basic validation
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred have different shapes: "
                f"{y_true.shape} vs {y_pred.shape}"
            )
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred must be non-negative."
            )
        if np.any(w < 0):
            raise ValueError(
                f"For entity {entity_id!r}, sample weights must be non-negative."
            )

        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)

        # If there is literally no error for this entity, we can just
        # pick R = 1.0 (or the closest in the grid) and under/over
        # costs will both be zero.
        if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
            R_e = float(ratios_arr[np.argmin(np.abs(ratios_arr - 1.0))])
            cu_e = R_e * co
            results.append(
                {
                    entity_col: entity_id,
                    "R": R_e,
                    "cu": cu_e,
                    "co": co,
                    "under_cost": 0.0,
                    "over_cost": 0.0,
                    "diff": 0.0,
                }
            )
            continue

        best_R = None
        best_cu = None
        best_under_cost = None
        best_over_cost = None
        best_diff = None

        for R in ratios_arr:
            cu_val = R * co

            under_cost = float(np.sum(w * cu_val * shortfall))
            over_cost = float(np.sum(w * co * overbuild))
            diff = abs(under_cost - over_cost)

            if (best_diff is None) or (diff < best_diff):
                best_diff = diff
                best_R = float(R)
                best_cu = float(cu_val)
                best_under_cost = under_cost
                best_over_cost = over_cost

        results.append(
            {
                entity_col: entity_id,
                "R": best_R,
                "cu": best_cu,
                "co": co,
                "under_cost": best_under_cost,
                "over_cost": best_over_cost,
                "diff": best_diff,
            }
        )

    return pd.DataFrame(results)