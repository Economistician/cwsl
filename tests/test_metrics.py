import os
import sys
import numpy as np
import pytest

# Ensure src/ is on the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import (
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


def test_cwsl_two_interval_example():
    y_true = [10, 12]
    y_pred = [8, 15]
    cu = 2.0
    co = 1.0

    value = cwsl(y_true, y_pred, cu=cu, co=co)
    assert np.isclose(value, 7.0 / 22.0)


def test_cwsl_simple_shortfall():
    """
    CWSL should match the hand-calculated value for a simple shortfall case.

    y_true = 100
    y_pred = 90
    cu = 3, co = 1

    shortfall = 10, overbuild = 0
    cost = 3 * 10 + 1 * 0 = 30
    demand = 100
    CWSL = 30 / 100 = 0.30
    """
    value = cwsl(y_true=[100], y_pred=[90], cu=3.0, co=1.0)
    assert np.isclose(value, 0.30)


def test_cwsl_simple_overbuild():
    """
    CWSL should match the hand-calculated value for a simple overbuild case.

    y_true = 100
    y_pred = 110
    cu = 3, co = 1

    shortfall = 0, overbuild = 10
    cost = 3 * 0 + 1 * 10 = 10
    demand = 100
    CWSL = 10 / 100 = 0.10
    """
    value = cwsl(y_true=[100], y_pred=[110], cu=3.0, co=1.0)
    assert np.isclose(value, 0.10)


def test_cwsl_symmetric_equals_wmape_scalar():
    """
    When cu == co and sample_weight is None, CWSL should equal wMAPE.
    This validates that in the symmetric case it behaves like a standard
    demand-normalized absolute error.
    """
    y_true = [100, 200, 50]
    y_pred = [90, 210, 40]

    cu = 1.0
    co = 1.0

    cwsl_val = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
    wmape_val = wmape(y_true=y_true, y_pred=y_pred, sample_weight=None)

    assert np.isclose(cwsl_val, wmape_val)


def test_cwsl_symmetric_equals_wmape_with_weights():
    """
    Same as above, but with non-uniform sample_weight to confirm the
    weighting logic aligns between CWSL and wMAPE.
    """
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([90.0, 210.0, 40.0])
    weights = np.array([1.0, 2.0, 0.5])

    cu = 1.0
    co = 1.0

    cwsl_val = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co, sample_weight=weights)
    wmape_val = wmape(y_true=y_true, y_pred=y_pred, sample_weight=weights)

    assert np.isclose(cwsl_val, wmape_val)


def test_cwsl_broadcasts_scalar_cu_co():
    """
    Scalar cu/co should broadcast correctly across multiple observations.

    obs 1: y=100, ŷ=90 → shortfall=10, overbuild=0, cost=2*10=20
    obs 2: y=50,  ŷ=60 → shortfall=0,  overbuild=10, cost=1*10=10
    total_cost = 30, total_demand = 150 → CWSL = 30/150 = 0.2
    """
    y_true = [100, 50]
    y_pred = [90, 60]

    cu = 2.0
    co = 1.0

    value = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
    assert np.isclose(value, 0.20)


def test_cwsl_zero_demand_zero_cost_returns_zero():
    """
    If total demand and total cost are both zero, CWSL is defined as 0.0.
    Example: no demand and perfect zero forecast.
    """
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    value = cwsl(y_true=y_true, y_pred=y_pred, cu=2.0, co=1.0)
    assert np.isclose(value, 0.0)


def test_cwsl_zero_demand_positive_cost_raises():
    """
    If total demand is zero but cost is positive, CWSL is undefined and
    should raise ValueError under this formulation.
    Example: y_true all zero, y_pred positive.
    """
    y_true = [0, 0]
    y_pred = [5, 10]

    with pytest.raises(ValueError):
        cwsl(y_true=y_true, y_pred=y_pred, cu=2.0, co=1.0)


def test_nsl_basic():
    y_true = [10, 12, 8]
    y_pred = [9, 15, 7]
    value = nsl(y_true, y_pred)
    assert np.isclose(value, 1.0 / 3.0)


def test_ud_basic():
    y_true = [10, 12, 8]
    y_pred = [8, 15, 5]
    value = ud(y_true, y_pred)
    assert np.isclose(value, 5.0 / 3.0)


def test_wmape_basic():
    y_true = [10, 20, 30]
    y_pred = [8, 20, 33]
    value = wmape(y_true, y_pred)
    assert np.isclose(value, 5.0 / 60.0)


def test_hr_at_tau_scalar_and_vector():
    y_true = [10, 12, 8]
    y_pred = [9, 15, 7]

    value_scalar = hr_at_tau(y_true, y_pred, tau=2.0)
    assert np.isclose(value_scalar, 2.0 / 3.0)

    value_vector = hr_at_tau(y_true, y_pred, tau=[0.5, 5.0, 0.5])
    assert np.isclose(value_vector, 1.0 / 3.0)


def test_frs_consistency():
    y_true = [10, 12, 8]
    y_pred = [9, 15, 7]
    cu = 2.0
    co = 1.0

    nsl_val = nsl(y_true, y_pred)
    cwsl_val = cwsl(y_true, y_pred, cu=cu, co=co)
    frs_val = frs(y_true, y_pred, cu=cu, co=co)

    assert np.isclose(frs_val, nsl_val - cwsl_val)


def test_mae_basic():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    value = mae(y_true, y_pred)
    assert np.isclose(value, 7.0 / 3.0)


def test_rmse_basic():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    expected = np.sqrt(17.0 / 3.0)
    value = rmse(y_true, y_pred)
    assert np.isclose(value, expected)


def test_mape_basic():
    y_true = [10, 20, 40]
    y_pred = [12, 18, 50]
    expected = (0.2 + 0.1 + 0.25) / 3.0
    value = mape(y_true, y_pred)
    assert np.isclose(value, expected)