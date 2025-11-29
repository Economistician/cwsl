import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is on the Python path (same pattern as your other tests)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import cwsl, evaluate_hierarchy_df


def test_evaluate_hierarchy_df_basic():
    # Simple toy dataset: 2 stores x 2 items x 1 interval each
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

    result = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # We should get one DataFrame per level
    assert set(result.keys()) == {"overall", "by_store", "by_store_item"}

    # ----- overall -----
    overall = result["overall"]
    # Single row
    assert len(overall) == 1
    # CWSL should match direct cwsl() call on the whole df
    y_true = df["actual_qty"].to_numpy(dtype=float)
    y_pred = df["forecast_qty"].to_numpy(dtype=float)
    expected_cwsl = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)

    assert np.isclose(overall["cwsl"].iloc[0], expected_cwsl)
    # Sanity checks on columns
    assert "n_intervals" in overall.columns
    assert "total_demand" in overall.columns
    assert "nsl" in overall.columns
    assert "frs" in overall.columns

    # ----- by_store -----
    by_store = result["by_store"]
    # Two stores: 1 and 2
    assert set(by_store["store_id"]) == {1, 2}
    # Required columns exist
    for col in ["n_intervals", "total_demand", "cwsl", "nsl", "ud", "wmape", "frs"]:
        assert col in by_store.columns

    # Each store should have at least one interval
    assert (by_store["n_intervals"] > 0).all()

    # ----- by_store_item -----
    by_store_item = result["by_store_item"]
    # 4 unique (store_id, item_id) combinations
    assert len(by_store_item) == 4
    assert set(by_store_item["store_id"]) == {1, 2}
    assert set(by_store_item["item_id"]) == {"A", "B"}