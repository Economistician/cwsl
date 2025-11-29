import os
import sys
import numpy as np
import pytest

# Ensure src/ is on the Python path (same pattern as your other tests)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import (
    cwsl,
    estimate_R_cost_balance,
    cwsl_sensitivity,
)


def test_cwsl_sensitivity_matches_direct_calls():
    y_true = [10, 12, 8]
    y_pred = [9, 15, 7]
    co = 1.0
    R_list = (0.5, 1.0, 2.0, 3.0)

    sens = cwsl_sensitivity(
        y_true=y_true,
        y_pred=y_pred,
        R_list=R_list,
        co=co,
    )

    # Keys should match R_list (as floats)
    assert set(sens.keys()) == set(float(r) for r in R_list)

    # Each sensitivity value should equal a direct cwsl call
    for R in R_list:
        cu = R * co
        val_direct = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
        assert np.isclose(sens[float(R)], val_direct)


def test_cwsl_sensitivity_rejects_empty_R_list():
    y_true = [10, 12]
    y_pred = [9, 11]

    with pytest.raises(ValueError):
        cwsl_sensitivity(
            y_true=y_true,
            y_pred=y_pred,
            R_list=(),
        )


def test_estimate_R_cost_balance_symmetric_pattern_prefers_one():
    """
    If the forecast is symmetrically off (same total shortfall and overbuild),
    R ≈ 1 should minimize the under/over cost gap when 1 is in the grid.
    """
    y_true = [10, 10]
    y_pred = [8, 12]  # shortfall = 2, overbuild = 2

    R_star = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert np.isclose(R_star, 1.0)


def test_estimate_R_cost_balance_ignores_non_positive_R():
    y_true = [10, 10]
    y_pred = [9, 11]

    R_star = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(-1.0, 0.0, 1.0, 2.0),
        co=1.0,
    )

    # Only positive R should matter; behavior should still be valid
    assert R_star in (1.0, 2.0)


def test_estimate_R_cost_balance_raises_if_no_valid_R():
    y_true = [10, 10]
    y_pred = [9, 11]

    with pytest.raises(ValueError):
        estimate_R_cost_balance(
            y_true=y_true,
            y_pred=y_pred,
            R_grid=(0.0, -1.0),  # no positive R
            co=1.0,
        )