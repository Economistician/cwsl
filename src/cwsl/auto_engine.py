from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .electric_barometer import (
    ElectricBarometer,
    LightGBMRegressorAdapter,
    CatBoostAdapter,
    ProphetAdapter,
    SarimaxAdapter,
    ArimaAdapter,
)

try:  # Optional XGBoost
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGB = True
except Exception:  # pragma: no cover - import guard
    HAS_XGB = False


class AutoEngine:
    """
    AutoEngine: convenience builder for ElectricBarometer model pools.

    It encapsulates a few common decisions:

      * Which model families to include (linear, trees, boosting, GBMs, time-series adapters).
      * Whether the problem is plain tabular vs univariate time series.
      * Whether optional dependencies (xgboost, lightgbm, catboost, prophet, statsmodels)
        are available, and only adds those engines if they are installed.

    Typical usage
    -------------
    >>> from cwsl import AutoEngine
    >>> ae = AutoEngine(
    ...     cu=2.0,
    ...     co=1.0,
    ...     use_linear=True,
    ...     use_trees=True,
    ...     use_xgboost=True,
    ... )
    >>> eb = ae.build_selector(X_train, y_train)
    >>> eb.fit(X_train, y_train, X_val, y_val)
    >>> print(eb.best_name_)

    Notes
    -----
    * Step 1 implementation focuses on **tabular** regression engines.
      Time-series adapters (Prophet, ARIMA/SARIMAX) are wired in but optional.
    """

    def __init__(
        self,
        *,
        cu: float = 2.0,
        co: float = 1.0,
        selection_mode: str = "holdout",
        cv: int = 3,
        random_state: Optional[int] = None,
        # Model family toggles
        use_dummy: bool = True,
        use_linear: bool = True,
        use_regularized_linear: bool = True,
        use_trees: bool = True,
        use_gbm: bool = True,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_catboost: bool = True,
        # Time-series style engines (future expansion / opt-in)
        use_prophet: bool = False,
        use_sarimax: bool = False,
        use_arima: bool = False,
    ) -> None:
        if cu <= 0 or co <= 0:
            raise ValueError("AutoEngine: cu and co must be strictly positive.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                "AutoEngine: selection_mode must be 'holdout' or 'cv', "
                f"got {selection_mode!r}."
            )

        self.cu = float(cu)
        self.co = float(co)
        self.selection_mode = selection_mode
        self.cv = int(cv)
        self.random_state = random_state

        # Model family flags
        self.use_dummy = bool(use_dummy)
        self.use_linear = bool(use_linear)
        self.use_regularized_linear = bool(use_regularized_linear)
        self.use_trees = bool(use_trees)
        self.use_gbm = bool(use_gbm)
        self.use_xgboost = bool(use_xgboost)
        self.use_lightgbm = bool(use_lightgbm)
        self.use_catboost = bool(use_catboost)

        self.use_prophet = bool(use_prophet)
        self.use_sarimax = bool(use_sarimax)
        self.use_arima = bool(use_arima)

    # ------------------------------------------------------------------
    # Internal: build tabular model zoo
    # ------------------------------------------------------------------
    def _build_tabular_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Construct a dictionary of model candidates for tabular regression.

        X and y are accepted so that future versions could adapt depth,
        n_estimators, etc., based on sample size or feature count.
        """
        models: Dict[str, Any] = {}

        n_samples, n_features = X.shape[0], X.shape[1] if X.ndim == 2 else (X.shape[0], 1)[1]

        # 1) Baseline
        if self.use_dummy:
            models["dummy_mean"] = DummyRegressor(strategy="mean")

        # 2) Linear family
        if self.use_linear:
            models["linear"] = LinearRegression()

        if self.use_regularized_linear:
            models["ridge"] = Ridge(alpha=1.0, random_state=self.random_state)
            models["lasso"] = Lasso(alpha=0.001, random_state=self.random_state)

        # 3) Tree ensembles
        if self.use_trees:
            models["rf"] = RandomForestRegressor(
                n_estimators=80,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            )

        if self.use_gbm:
            models["gbr"] = GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                random_state=self.random_state,
            )

        # 4) Gradient-boosting libraries (optional)
        if self.use_xgboost and HAS_XGB:
            models["xgb"] = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=120,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
            )

        if self.use_lightgbm:
            try:
                models["lgbm"] = LightGBMRegressorAdapter(
                    n_estimators=150,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.random_state,
                )
            except Exception:
                # LightGBM not installed; silently skip
                pass

        if self.use_catboost:
            try:
                models["catboost"] = CatBoostAdapter(
                    iterations=200,
                    depth=4,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=self.random_state,
                )
            except Exception:
                # catboost not installed; silently skip
                pass

        return models

    # ------------------------------------------------------------------
    # Public API: build ElectricBarometer
    # ------------------------------------------------------------------
    def build_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        problem_type: str = "tabular",
    ) -> Dict[str, Any]:
        """
        Build the dictionary of candidate models for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features. For now, only used for tabular problems.

        y : array-like, shape (n_samples,)
            Training targets.

        problem_type : {"tabular", "univariate_ts"}, default "tabular"
            - "tabular": treat X as regular feature matrix, use ML regressors.
            - "univariate_ts": future hook for ARIMA/Prophet-style auto pools.

        Returns
        -------
        models : dict[str, Any]
            Model name → estimator/adapter with fit/predict.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if problem_type == "tabular":
            return self._build_tabular_models(X_arr, y_arr)

        if problem_type == "univariate_ts":
            # Placeholder: for now just fall back to tabular pool on shape.
            # A future version can build a time-series-specific zoo here.
            return self._build_tabular_models(X_arr.reshape(-1, 1), y_arr)

        raise ValueError(
            "AutoEngine.build_models: problem_type must be 'tabular' or 'univariate_ts', "
            f"got {problem_type!r}."
        )

    def build_selector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        problem_type: str = "tabular",
    ) -> ElectricBarometer:
        """
        Build an ElectricBarometer configured with an auto-constructed model zoo.

        This does **not** call .fit() — it just returns the selector ready
        to be fit on your preferred train/validation split.

        Example
        -------
        >>> ae = AutoEngine(cu=2.0, co=1.0)
        >>> eb = ae.build_selector(X_train, y_train)
        >>> eb.fit(X_train, y_train, X_val, y_val)
        """
        models = self.build_models(X, y, problem_type=problem_type)

        if not models:
            raise RuntimeError(
                "AutoEngine.build_selector constructed an empty model dictionary. "
                "Check your model-family flags or optional dependency installs."
            )

        eb = ElectricBarometer(
            models=models,
            cu=self.cu,
            co=self.co,
            selection_mode=self.selection_mode,
            cv=self.cv,
            random_state=self.random_state,
        )
        return eb

    def __repr__(self) -> str:
        return (
            "AutoEngine("
            f"cu={self.cu}, co={self.co}, "
            f"selection_mode={self.selection_mode!r}, cv={self.cv}, "
            f"use_dummy={self.use_dummy}, use_linear={self.use_linear}, "
            f"use_regularized_linear={self.use_regularized_linear}, "
            f"use_trees={self.use_trees}, use_gbm={self.use_gbm}, "
            f"use_xgboost={self.use_xgboost}, "
            f"use_lightgbm={self.use_lightgbm}, use_catboost={self.use_catboost}, "
            f"use_prophet={self.use_prophet}, use_sarimax={self.use_sarimax}, "
            f"use_arima={self.use_arima})"
        )