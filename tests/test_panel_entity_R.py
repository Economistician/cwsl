from __future__ import annotations

import numpy as np
import pandas as pd

from cwsl import (
    cwsl,
    nsl,
    estimate_entity_R_from_balance,
    evaluate_panel_with_entity_R,
)


def _build_toy_panel() -> pd.DataFrame:
    rows = []

    # Entity A: mostly over-forecast (tends to have overbuild)
    for t, y in enumerate([10, 12, 15, 20], start=1):
        rows.append(
            {
                "entity": "A_over_lean",
                "t": t,
                "y_true": y,
                "y_pred": y + (2 if t % 2 == 0 else 1),
            }
        )

    # Entity B: roughly balanced around the truth
    for t, y in enumerate([10, 12, 15, 20], start=1):
        bias = 1 if t % 2 == 0 else -1
        rows.append(
            {
                "entity": "B_balanced",
                "t": t,
                "y_true": y,
                "y_pred": y + bias,
            }
        )

    # Entity C: mostly under-forecast (tends to have shortfall)
    for t, y in enumerate([10, 12, 15, 20], start=1):
        rows.append(
            {
                "entity": "C_under_lean",
                "t": t,
                "y_true": y,
                "y_pred": y - (2 if t % 2 == 0 else 1),
            }
        )

    return pd.DataFrame(rows)


def test_evaluate_panel_with_entity_R_basic_consistency():
    df = _build_toy_panel()

    ratios = (0.5, 1.0, 2.0, 3.0)
    co = 1.0

    entity_R = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="y_true",
        y_pred_col="y_pred",
        ratios=ratios,
        co=co,
        sample_weight_col=None,
    )

    summary = evaluate_panel_with_entity_R(
        df=df,
        entity_R=entity_R,
        entity_col="entity",
        y_true_col="y_true",
        y_pred_col="y_pred",
        tau=2.0,
        sample_weight_col=None,
    )

    # We expect one row per entity
    assert len(summary) == entity_R.shape[0]

    expected_cols = {
        "entity",
        "R",
        "cu",
        "co",
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
    assert set(summary.columns) == expected_cols

    # For each entity, CWSL in the summary should match a direct call using
    # that entity's R and co.
    for _, row in summary.iterrows():
        entity_id = row["entity"]

        g = df[df["entity"] == entity_id]
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)

        ent_row = entity_R[entity_R["entity"] == entity_id].iloc[0]
        R_e = float(ent_row["R"])
        co_e = float(ent_row["co"])
        cu_e = R_e * co_e

        cwsl_direct = cwsl(
            y_true=y_true,
            y_pred=y_pred,
            cu=cu_e,
            co=co_e,
        )

        assert np.isclose(row["CWSL"], cwsl_direct)

    # NSL for one entity should also match direct computation
    ent = "B_balanced"
    g = df[df["entity"] == ent]
    y_true = g["y_true"].to_numpy(dtype=float)
    y_pred = g["y_pred"].to_numpy(dtype=float)

    row_B = summary[summary["entity"] == ent].iloc[0]
    nsl_direct = nsl(y_true=y_true, y_pred=y_pred)
    assert np.isclose(row_B["NSL"], nsl_direct)