import numpy as np
import pandas as pd

from cwsl import compare_forecasts, cwsl, nsl, frs

def test_compare_forecasts_basic():
    y_true = np.array([10, 12, 8])

    forecasts = {
        "under": np.array([9, 11, 7]),
        "over":  np.array([12, 14, 10]),
    }

    cu = 2.0
    co = 1.0

    df = compare_forecasts(
        y_true=y_true,
        forecasts=forecasts,
        cu=cu,
        co=co,
        tau=2,
    )

    # Should have 2 rows (one per model)
    assert len(df) == 2

    # Key metric sanity checks
    assert "CWSL" in df.columns
    assert "NSL" in df.columns
    assert "FRS" in df.columns

    # Compare values match individual function calls
    assert df.loc["under", "CWSL"] == cwsl(y_true, forecasts["under"], cu=cu, co=co)
    assert df.loc["under", "NSL"] == nsl(y_true, forecasts["under"])
    assert df.loc["under", "FRS"] == frs(y_true, forecasts["under"], cu=cu, co=co)