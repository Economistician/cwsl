from __future__ import annotations

import numpy as np
import pandas as pd

from cwsl import (
    cwsl,
    compute_cwsl_df,
    nsl,
    ud,
    wmape,
    hr_at_tau,
    frs,
    mae,
    rmse,
    mape,
    evaluate_groups_df,
)


def test_compute_cwsl_df_scalar_cu_co_matches_core():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
        }
    )

    cu = 2.0
    co = 1.0

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=cu,
        co=co,
    )

    via_df = compute_cwsl_df(
        df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu=cu,
        co=co,
    )

    assert np.isclose(direct, via_df)


def test_compute_cwsl_df_column_cu_co_matches_core():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
            "cu_col": [2.0, 2.0, 2.0],
            "co_col": [1.0, 1.0, 1.0],
        }
    )

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=df["cu_col"].to_numpy(),
        co=df["co_col"].to_numpy(),
    )

    via_df = compute_cwsl_df(
        df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu="cu_col",
        co="co_col",
    )

    assert np.isclose(direct, via_df)


def test_compute_cwsl_df_with_sample_weight():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    cu = 2.0
    co = 1.0

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=cu,
        co=co,
        sample_weight=df["weight"].to_numpy(),
    )

    via_df = compute_cwsl_df(
        df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu=cu,
        co=co,
        sample_weight_col="weight",
    )

    assert np.isclose(direct, via_df)


def test_evaluate_groups_df_matches_direct_metrics():
    df = pd.DataFrame(
        {
            "store_id": [1, 1, 1, 2, 2, 2],
            "item_id": ["A", "A", "B", "A", "A", "B"],
            "actual_qty": [10, 12, 8, 9, 11, 7],
            "forecast_qty": [9, 15, 7, 10, 10, 8],
        }
    )

    cu = 2.0
    co = 1.0
    tau = 2.0

    summary = evaluate_groups_df(
        df,
        group_cols=["store_id", "item_id"],
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # We expect one row per (store_id, item_id)
    assert set(summary.columns) == {
        "store_id",
        "item_id",
        "CWSL",
        "NSL",
        "UD",
        "wMAPE",
        "HR@tau",
        "FRS",
        "MAE",
        "RMSE",
        "MAPE",
    }
    assert len(summary) == 4

    # Pick one group and compare to direct calls
    g = df[(df["store_id"] == 1) & (df["item_id"] == "A")]
    y_true = g["actual_qty"].to_numpy()
    y_pred = g["forecast_qty"].to_numpy()

    expected = {
        "CWSL": cwsl(y_true, y_pred, cu=cu, co=co),
        "NSL": nsl(y_true, y_pred),
        "UD": ud(y_true, y_pred),
        "wMAPE": wmape(y_true, y_pred),
        "HR@tau": hr_at_tau(y_true, y_pred, tau=tau),
        "FRS": frs(y_true, y_pred, cu=cu, co=co),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

    row = summary[
        (summary["store_id"] == 1) & (summary["item_id"] == "A")
    ].iloc[0]

    for name, val in expected.items():
        assert np.isclose(row[name], val)


def test_evaluate_groups_df_with_per_row_cu_co():
    """
    Ensure evaluate_groups_df can consume per-row cu/co (via column names)
    and that a group with more expensive shortfalls ends up with a higher
    CWSL than one with cheaper shortfalls, all else equal.
    """
    rows = []

    # Entity A: always over-forecast (more overbuild), cheap shortfall
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "A",
                "t": t,
                "actual_qty": y,
                "forecast_qty": y + 2,  # always a bit high
                "cu": 1.0,              # shortfall cost (cheap)
                "co": 1.0,
            }
        )

    # Entity B: always under-forecast (more shortfall), expensive shortfall
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "B",
                "t": t,
                "actual_qty": y,
                "forecast_qty": y - 2,  # always a bit low
                "cu": 3.0,              # shortfall cost (expensive)
                "co": 1.0,
            }
        )

    df = pd.DataFrame(rows)

    result = evaluate_groups_df(
        df=df,
        group_cols=["entity"],
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu="cu",   # per-row cu
        co="co",   # per-row co
        tau=2.0,
    )

    # We expect one row per entity
    assert set(result["entity"]) == {"A", "B"}

    # CWSL should be higher for entity B, where shortfalls are more expensive
    cwsl_A = float(result.loc[result["entity"] == "A", "CWSL"].iloc[0])
    cwsl_B = float(result.loc[result["entity"] == "B", "CWSL"].iloc[0])

    assert cwsl_B > cwsl_A