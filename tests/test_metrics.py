import os
import sys
import numpy as np

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


def test_cwsl_basic():
    y_true = [10, 12]
    y_pred = [8, 15]
    cu = 2.0
    co = 1.0

    value = cwsl(y_true, y_pred, cu=cu, co=co)
    assert np.isclose(value, 7.0 / 22.0)


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