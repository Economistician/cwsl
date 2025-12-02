from __future__ import annotations

from .metrics import (
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

from .dataframe import (
    compute_cwsl_df, 
    evaluate_groups_df, 
    evaluate_hierarchy_df,
    evaluate_panel_df,
    evaluate_panel_with_entity_R,
)

from .compare import (
    compare_forecasts, 
    select_model_by_cwsl,
)

from .scorers import (
    cwsl_loss,
    cwsl_scorer,
)

from .costs import (
    estimate_R_cost_balance,
    estimate_entity_R_from_balance,
)

from .sensitivity import cwsl_sensitivity

from .training import make_cwsl_keras_loss

from .electric_barometer import (
    ElectricBarometer, 
    BaseAdapter,
    ProphetAdapter,
    SarimaxAdapter,
    ArimaAdapter,
    LightGBMRegressorAdapter,
    CatBoostAdapter,
)

from .cwsl_regressor import CWSLRegressor

from .auto_engine import AutoEngine

__all__ = [
    # Core metrics
    "cwsl",
    "nsl",
    "ud",
    "wmape",
    "hr_at_tau",
    "frs",
    "mae",
    "rmse",
    "mape",

    # DataFrame utilities
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",

    # Comparison utilities
    "compare_forecasts",
    "select_model_by_cwsl",
    "estimate_R_cost_balance",
    "estimate_entity_R_from_balance",

    # Sensitivity analysis
    "cwsl_sensitivity",

    # Scikit-learn scorers & loss
    "cwsl_loss",
    "cwsl_scorer",

    # Keras loss
    "make_cwsl_keras_loss",

    # High-level engine
    "ElectricBarometer",
    "BaseAdapter",
    "ProphetAdapter",
    "SarimaxAdapter",
    "ArimaAdapter",
    "LightGBMRegressorAdapter",
    "CatBoostAdapter",

    "CWSLRegressor",

    "AutoEngine"
]