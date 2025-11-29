from __future__ import annotations

from typing import Dict, Iterable, Sequence, Union

import numpy as np

from .metrics import cwsl

ArrayLike = Union[Iterable[float], np.ndarray]


def cwsl_sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, ArrayLike] = 1.0,
    sample_weight: ArrayLike | None = None,
) -> Dict[float, float]:
    """
    Cost Sensitivity Analysis for CWSL.

    Evaluate the Cost-Weighted Service Loss (CWSL) across a range of
    cost ratios R = cu / co, holding co fixed and setting cu = R * co
    for each R in R_list.

    This is the core building block for "Cost Sensitivity Analysis":
    it lets you see how sensitive a model's performance is to different
    assumptions about the shortfall-vs-overbuild cost ratio.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand per interval. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand per interval. Must be non-negative.

    R_list : sequence of float, default=(0.5, 1.0, 2.0, 3.0)
        Candidate cost ratios R = cu / co to evaluate.
        - R = 1.0  → symmetric costs (short and long equally bad)
        - R > 1.0  → shortfall is R times worse than overbuild
        - R < 1.0  → overbuild is 1/R times worse than shortfall

        By default, we evaluate {0.5, 1.0, 2.0, 3.0}, which covers:
        - mild "overbuild worse" (0.5),
        - symmetric (1.0),
        - moderate and strong shortfall asymmetry (2.0, 3.0).

    co : float or array-like of shape (n_samples,), default=1.0
        Overbuild cost per unit. Can be:
        - scalar: same overbuild cost for all intervals;
        - 1D array: per-interval overbuild cost.

        For each R in R_list, we set cu = R * co and compute CWSL.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. Passed directly
        into `cwsl`. See `cwsl` docstring for details.

    Returns
    -------
    Dict[float, float]
        A dictionary mapping each R in R_list to its corresponding
        CWSL score:

            { R_1: CWSL(R_1), R_2: CWSL(R_2), ... }

        Only strictly positive R values are used; non-positive R
        values in R_list are ignored. If no valid R remains, a
        ValueError is raised.

    Raises
    ------
    ValueError
        If R_list is empty or contains no positive values, or if
        `cwsl` raises due to invalid data (e.g., negative demand).
    """
    R_arr = np.asarray(R_list, dtype=float)
    if R_arr.ndim != 1 or R_arr.size == 0:
        raise ValueError("R_list must be a non-empty 1D sequence of floats.")

    results: Dict[float, float] = {}

    for R in R_arr:
        if R <= 0:
            # skip non-positive ratios; they are not meaningful as cu / co
            continue

        cu = R * co
        value = cwsl(
            y_true=y_true,
            y_pred=y_pred,
            cu=cu,
            co=co,
            sample_weight=sample_weight,
        )
        results[float(R)] = float(value)

    if not results:
        raise ValueError(
            "No valid R values in R_list (must contain at least one positive value)."
        )

    return results