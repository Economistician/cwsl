import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is on the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import cwsl, evaluate_panel_df


def test_evaluate_panel_df_long_format():
    # Toy dataset: 2 stores x 2 items x 1 interval each
    df = pd.DataFrame(
        {
            "store_id": [1, 1, 2, 2],
            "item_id": ["A", "B", "A", "B"],
            "actual_qty": [10, 12, 8, 9],
            "forecast_qty": [9, 11, 7, 10],
        }
    )

    levels = {
        "overall": [],
        "by_store": ["store_id"],
        "by_store_item": ["store_id", "item_id"],
    }

    cu = 2.0
    co = 1.0
    tau = 2.0

    panel = evaluate_panel_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # Basic shape checks
    assert "level" in panel.columns
    assert "metric" in panel.columns
    assert "value" in panel.columns

    # Levels present as expected
    assert set(panel["level"]) == {"overall", "by_store", "by_store_item"}

    # There should be at least cwsl and n_intervals metrics
    assert {"cwsl", "n_intervals"}.issubset(set(panel["metric"].unique()))

    # For the 'overall' + 'cwsl' row, value should match direct cwsl() on entire df
    y_true = df["actual_qty"].to_numpy(dtype=float)
    y_pred = df["forecast_qty"].to_numpy(dtype=float)
    expected_cwsl = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)

    overall_cwsl_rows = panel[
        (panel["level"] == "overall") & (panel["metric"] == "cwsl")
    ]
    # Should be exactly one such row
    assert len(overall_cwsl_rows) == 1

    actual_value = float(overall_cwsl_rows["value"].iloc[0])
    assert np.isclose(actual_value, expected_cwsl)