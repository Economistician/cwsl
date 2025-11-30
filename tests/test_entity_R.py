import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is on the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from cwsl import estimate_entity_R_from_balance


def test_estimate_entity_R_from_balance_basic_shape():
    # Construct a tiny panel with three entities that have different error patterns
    rows = []

    # Entity A: mostly over-forecast (tends to have overbuild)
    for t, y in enumerate([10, 12, 15, 20], start=1):
        rows.append(
            {
                "entity": "A_over_lean",
                "t": t,
                "y_true": y,
                "y_pred": y + (2 if t % 2 == 0 else 1),  # always a bit high
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
                "y_pred": y + bias,  # sometimes low, sometimes high
            }
        )

    # Entity C: mostly under-forecast (tends to have shortfall)
    for t, y in enumerate([10, 12, 15, 20], start=1):
        rows.append(
            {
                "entity": "C_under_lean",
                "t": t,
                "y_true": y,
                "y_pred": y - (2 if t % 2 == 0 else 1),  # always a bit low
            }
        )

    df = pd.DataFrame(rows)

    ratios = (0.5, 1.0, 2.0, 3.0)
    co = 1.0

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="y_true",
        y_pred_col="y_pred",
        ratios=ratios,
        co=co,
        sample_weight_col=None,
    )

    # Expect one row per entity
    assert set(result["entity"]) == {"A_over_lean", "B_balanced", "C_under_lean"}

    # Columns should include the expected fields
    for col in ["R", "cu", "co", "under_cost", "over_cost", "diff"]:
        assert col in result.columns

    # R should always come from the provided grid
    for R_val in result["R"]:
        assert R_val in ratios

    # cu should equal R * co for each row
    for _, row in result.iterrows():
        assert np.isclose(row["cu"], row["R"] * row["co"])