from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .compare import select_model_by_cwsl
from .metrics import cwsl, rmse, wmape


def _clone_model(model: Any) -> Any:
    """
    Lightweight clone for sklearn-style estimators or adapter-like objects.

    Cloning strategy
    ----------------
    1. If sklearn is available, try sklearn.base.clone(model).
    2. Otherwise, if the object implements get_params(), re-instantiate via:
           model.__class__(**model.get_params())
    3. As a last resort, call model.__class__() with no arguments.

    Notes for custom adapters
    -------------------------
    If you implement your own adapter (e.g., for statsmodels, Prophet, ARIMA),
    you have two main options to play nicely with ElectricBarometer:

    - Make your adapter "configuration-only" at __init__ time and implement
      get_params() so that it can be reconstructed via __class__(**params).

    - Or, if you need custom cloning logic, instantiate fresh adapters in your
      own code before passing them into ElectricBarometer.
    """
    # Try sklearn.clone if available
    try:
        from sklearn.base import clone  # type: ignore

        return clone(model)
    except Exception:
        pass

    # Fallback: re-create via class + get_params
    if hasattr(model, "get_params"):
        params = model.get_params()
        return model.__class__(**params)

    # Last resort: call class with no args
    return model.__class__()


class BaseAdapter:
    """
    Minimal base class for non-sklearn-style forecast engines.

    Subclasses are expected to wrap libraries like statsmodels, Prophet,
    ARIMA/SARIMAX, or any custom forecasting engine, and present a
    scikit-learn-like interface:

        adapter = MyAdapter(...)
        adapter.fit(X, y)       # returns self
        y_pred = adapter.predict(X)

    Requirements
    ------------
    - fit(self, X, y, sample_weight=None) -> BaseAdapter
        X : array-like or ignored (for pure time-series models)
        y : 1D array-like of targets
        sample_weight : optional, can be ignored if not supported.

    - predict(self, X) -> array-like
        Should return a 1D numpy array of predictions for each row in X.

    ElectricBarometer itself does *not* need to know whether a model is a
    "real" sklearn estimator or an adapter; it simply calls .fit() and
    .predict(). This base class is provided as a clear, documented contract
    for users who want to adapt non-sklearn libraries into the EB ecosystem.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseAdapter":
        raise NotImplementedError(
            "BaseAdapter subclasses must implement .fit(X, y, sample_weight=None)."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "BaseAdapter subclasses must implement .predict(X)."
        )


class ProphetAdapter(BaseAdapter):
    """
    Adapter for `prophet.Prophet` so it can be used inside ElectricBarometer.

    Assumptions
    -----------
    * X encodes the time index:
        - shape (n_samples,) of datetime-like values, or
        - shape (n_samples, n_features) where the *first column* is datetime-like.
    * y is a 1D array-like of numeric targets (demand, load, etc).

    Behavior
    --------
    - At fit-time, constructs a DataFrame with columns:
          ds = timestamps from X
          y  = targets from y
      and calls `Prophet.fit(df)`.

    - At predict-time, constructs a DataFrame with:
          ds = timestamps from X
      and returns the Prophet `yhat` column as a 1D numpy array.

    Optional dependency
    -------------------
    Prophet is not required for CWSL itself. If the `prophet` package is not
    installed, constructing ProphetAdapter() will raise an ImportError with a
    helpful message.

    You can also pass an already-configured Prophet instance:

        from prophet import Prophet
        base = Prophet(...)
        adapter = ProphetAdapter(model=base)
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        if model is None:
            try:
                from prophet import Prophet as _Prophet  # type: ignore
            except Exception as e:  # pragma: no cover - import failure path
                raise ImportError(
                    "ProphetAdapter requires the optional 'prophet' package. "
                    "Install it via `pip install prophet`."
                ) from e

            model = _Prophet()

        self.model = model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,  # ignored
    ) -> "ProphetAdapter":
        # Local import to avoid hard dependency at module import time
        import pandas as _pd

        X_arr = np.asarray(X)
        # Use the first column if 2D
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = _pd.to_datetime(X_arr)
        y_arr = np.asarray(y, dtype=float)

        df = _pd.DataFrame({"ds": ds, "y": y_arr})
        self.model.fit(df)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as _pd

        X_arr = np.asarray(X)
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = _pd.to_datetime(X_arr)
        df_future = _pd.DataFrame({"ds": ds})
        forecast = self.model.predict(df_future)

        if "yhat" not in forecast.columns:
            raise RuntimeError(
                "ProphetAdapter: expected 'yhat' column in forecast output."
            )

        return np.asarray(forecast["yhat"], dtype=float)


# Optional statsmodels / SARIMAX / ARIMA support
try:  # pragma: no cover - import guard
    import statsmodels.api as _sm  # type: ignore[import]

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - import guard
    _sm = None
    HAS_STATSMODELS = False


class SarimaxAdapter(BaseAdapter):
    """
    Adapter for statsmodels SARIMAX to make it EB-compatible.

    This wrapper:

      - Ignores X entirely (as is typical for univariate SARIMAX).
      - Fits on y as a simple time series.
      - On predict(X_val), it forecasts len(X_val) steps ahead from the end
        of the training series.

    This is enough to let ElectricBarometer perform holdout or CV selection
    *without* EB needing to know anything about statsmodels.

    Parameters
    ----------
    order : tuple, default (1, 0, 0)
        ARIMA (p, d, q) order.

    seasonal_order : tuple, default (0, 0, 0, 0)
        Seasonal (P, D, Q, s) order.

    trend : str or None, default None
        Trend parameter passed to SARIMAX.

    enforce_stationarity, enforce_invertibility : bool, default True
        Passed through to SARIMAX.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self._result = None

        if not HAS_STATSMODELS:
            # We delay the actual failure until fit(), but this is a clear signal.
            # Users can skip tests or avoid instantiating this adapter if they
            # don't have statsmodels installed.
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SarimaxAdapter":
        if not HAS_STATSMODELS:
            raise ImportError(
                "SarimaxAdapter requires 'statsmodels' to be installed. "
                "Install it with `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Basic univariate SARIMAX on y only.
        model = _sm.tsa.statespace.SARIMAX(
            y_arr,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        # Use a small maxiter to keep tests lightweight.
        self._result = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError(
                "SarimaxAdapter has not been fit yet. Call .fit(X, y) first."
            )

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        # Forecast n_steps ahead from the end of the training sample.
        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    # Minimal param API so sklearn.clone or our _clone_model can reconstruct it.
    def get_params(self, deep: bool = True) -> dict:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return (
            f"SarimaxAdapter(order={self.order}, "
            f"seasonal_order={self.seasonal_order}, trend={self.trend!r})"
        )


class ArimaAdapter(BaseAdapter):
    """
    Adapter for statsmodels ARIMA to make it EB-compatible.

    This wrapper:

      - Ignores X entirely (univariate time-series model).
      - Fits on y as a simple ARIMA(p, d, q) series.
      - On predict(X_val), it forecasts len(X_val) steps ahead from the end
        of the training series.

    That is enough for ElectricBarometer to perform holdout or CV selection
    without knowing anything about statsmodels itself.

    Parameters
    ----------
    order : tuple, default (1, 0, 0)
        ARIMA (p, d, q) order.

    trend : str or None, default None
        Trend parameter passed to statsmodels.tsa.ARIMA.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        trend: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self._result = None

        if not HAS_STATSMODELS:
            # Delay failure until fit(), but this flag controls tests.
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ArimaAdapter":
        if not HAS_STATSMODELS:
            raise ImportError(
                "ArimaAdapter requires 'statsmodels' to be installed. "
                "Install it with `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Classic ARIMA interface
        model = _sm.tsa.ARIMA(
            y_arr,
            order=self.order,
            trend=self.trend,
        )

        # Default fit is fine for synthetic / test use
        self._result = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError(
                "ArimaAdapter has not been fit yet. Call .fit(X, y) first."
            )

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    # Minimal param API so _clone_model() can reconstruct it.
    def get_params(self, deep: bool = True) -> dict:
        return {
            "order": self.order,
            "trend": self.trend,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return f"ArimaAdapter(order={self.order}, trend={self.trend!r})"


# Optional CatBoost support
try:  # pragma: no cover - import guard
    from catboost import CatBoostRegressor as _CatBoostRegressor  # type: ignore[import]

    HAS_CATBOOST = True
except Exception:  # pragma: no cover - import guard
    _CatBoostRegressor = None
    HAS_CATBOOST = False


class CatBoostAdapter(BaseAdapter):
    """
    Adapter for `catboost.CatBoostRegressor` so it can be used inside
    ElectricBarometer with optional dependency handling and clean cloning.

    Notes
    -----
    * X and y are treated as regular tabular regression data.
    * sample_weight is passed through when provided.
    * Verbose training output is disabled by default to keep logs clean.

    Example
    -------
    models = {
        "dummy": DummyRegressor(strategy="mean"),
        "catboost": CatBoostAdapter(
            depth=4,
            learning_rate=0.1,
            iterations=200,
            loss_function="RMSE",
        ),
    }
    eb = ElectricBarometer(models=models, cu=2.0, co=1.0)
    eb.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, **params: Any) -> None:
        if not HAS_CATBOOST:
            raise ImportError(
                "CatBoostAdapter requires the optional 'catboost' package. "
                "Install it via `pip install catboost`."
            )
        self._params: Dict[str, Any] = dict(params)
        # Default quiet training if user didn't override verbosity
        if "verbose" not in self._params:
            self._params["verbose"] = False
        self._model: Optional[_CatBoostRegressor] = _CatBoostRegressor(
            **self._params
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CatBoostAdapter":
        if not HAS_CATBOOST or self._model is None:
            raise RuntimeError(
                "CatBoostAdapter cannot fit because 'catboost' is not available "
                "or the internal model is not initialized."
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            self._model.fit(X_arr, y_arr, sample_weight=sw_arr)
        else:
            self._model.fit(X_arr, y_arr)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "CatBoostAdapter has not been fit yet. Call .fit(X, y) first."
            )

        X_arr = np.asarray(X)
        y_pred = self._model.predict(X_arr)
        return np.asarray(y_pred, dtype=float).ravel()

    # Minimal param API so sklearn.clone or our _clone_model can reconstruct it.
    def get_params(self, deep: bool = True) -> dict:
        return dict(self._params)

    def set_params(self, **params):
        self._params.update(params)
        if HAS_CATBOOST:
            self._model = _CatBoostRegressor(**self._params)
        return self

    def __repr__(self) -> str:
        return f"CatBoostAdapter(params={self._params})"


class LightGBMRegressorAdapter(BaseAdapter):
    """
    Adapter for `lightgbm.LGBMRegressor` so it can be used inside ElectricBarometer.

    - Lazily imports LightGBM (no hard dependency for core CWSL).
    - Stores init kwargs so `_clone_model` can reconstruct via `__class__(**params)`.

    Typical usage
    -------------
    >>> from cwsl import LightGBMRegressorAdapter
    >>> models = {
    ...     "lgbm": LightGBMRegressorAdapter(
    ...         n_estimators=200,
    ...         learning_rate=0.05,
    ...         max_depth=-1,
    ...     ),
    ... }
    """

    def __init__(self, **lgbm_kwargs: Any) -> None:
        try:
            from lightgbm import LGBMRegressor  # type: ignore
        except Exception as e:  # pragma: no cover - import failure path
            raise ImportError(
                "LightGBMRegressorAdapter requires the optional 'lightgbm' "
                "package. Install it via `pip install lightgbm`."
            ) from e

        self._LGBMRegressor = LGBMRegressor
        self.lgbm_kwargs: Dict[str, Any] = dict(lgbm_kwargs)
        self.model = LGBMRegressor(**self.lgbm_kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LightGBMRegressorAdapter":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            self.model.fit(X_arr, y_arr, sample_weight=sw)
        else:
            self.model.fit(X_arr, y_arr)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float)

    # sklearn-like param API so _clone_model can reconstruct via __class__(**params)
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.lgbm_kwargs)

    def set_params(self, **params: Any) -> "LightGBMRegressorAdapter":
        self.lgbm_kwargs.update(params)
        # Rebuild underlying model with updated kwargs
        self.model = self._LGBMRegressor(**self.lgbm_kwargs)
        return self


class ElectricBarometer:
    """
    ElectricBarometer: cost-aware model selector built on CWSL.

    This is a high-level wrapper that:

      * Takes a dictionary of candidate forecast models (typically scikit-learn
        regressors with .fit() / .predict() methods, or adapters that follow
        the same interface).
      * Trains all candidates either:
          - on a holdout split (train/validation), or
          - via simple K-fold cross-validation (CV).
      * Evaluates them using:
          - CWSL (with your cu/co),
          - plus reference metrics (RMSE, wMAPE).
      * Selects the winner by **minimizing CWSL**.
      * Optionally refits the winning model on all available data
        (train ∪ validation) in holdout mode.
      * Exposes a clean .fit() / .predict() API and a results_ DataFrame.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of candidate models. Keys are model names, values are
        estimator objects with scikit-learn style API:

            model.fit(X_train, y_train, sample_weight=...)
            model.predict(X_val)

        You can also pass custom adapters that subclass `BaseAdapter` or at
        least implement the same .fit/.predict interface.

    cu : float, default 2.0
        Underbuild (shortfall) cost per unit. Must be strictly positive.

    co : float, default 1.0
        Overbuild (excess) cost per unit. Must be strictly positive.

    tau : float, default 2.0
        Reserved for future diagnostics (e.g., HR@τ) that may be attached
        to the ElectricBarometer workflow.

    training_mode : {"selection_only"}, default "selection_only"
        Reserved for future extension. In v0.3.x, only "selection_only"
        is supported (models train with their own objective; CWSL is used
        only for validation-time selection).

    refit_on_full : bool, default False
        In **holdout** mode:
            If True, after selecting the best model by CWSL on the validation
            set, refit that winning model on the concatenated (train ∪ val)
            data before exposing it via .best_model_ and .predict().

        In **cv** mode:
            X_train is treated as the full dataset; the winner is always
            refit on X_train, so this flag has no additional effect.

    selection_mode : {"holdout", "cv"}, default "holdout"
        - "holdout": use the provided (X_train, y_train, X_val, y_val) split
          and call `select_model_by_cwsl`.
        - "cv": ignore X_val / y_val and perform K-fold cross-validation on
          X_train / y_train to select the best model by mean CWSL.

    cv : int, default 3
        Number of folds to use when selection_mode="cv".
        Must be at least 2 in CV mode.

    random_state : int or None, default None
        Seed for the random number generator used in CV splitting.

    Attributes
    ----------
    best_name_ : str or None
        Name of the selected best model after .fit().

    best_model_ : Any or None
        The selected model object itself (fitted). In holdout mode, this may
        be either the model fitted on the validation split or a refit on
        (train ∪ val), depending on refit_on_full. In cv mode, it is the
        winner refit on all of X_train.

    results_ : pandas.DataFrame or None
        - In holdout mode: comparison table returned by `select_model_by_cwsl`,
          with one row per candidate model (index = model name).
        - In cv mode: table of mean CV scores per model (index = model name),
          with columns CWSL, RMSE, wMAPE.

    validation_cwsl_ : float or None
        Validation CWSL of the winning model:
        - In holdout mode: CWSL on the validation set.
        - In cv mode: mean CWSL across CV folds.

    validation_rmse_ : float or None
        RMSE value of the winning model on validation:
        - In holdout mode: from `select_model_by_cwsl`, if available.
        - In cv mode: mean CV RMSE.

    validation_wmape_ : float or None
        wMAPE value of the winning model on validation:
        - In holdout mode: from `select_model_by_cwsl`, if available.
        - In cv mode: mean CV wMAPE.

    r_ : float
        Cost ratio R = cu / co used for selection.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        training_mode: str = "selection_only",
        refit_on_full: bool = False,
        selection_mode: str = "holdout",
        cv: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        if not models:
            raise ValueError("ElectricBarometer requires at least one candidate model.")

        if training_mode != "selection_only":
            raise ValueError(
                "In v0.3.x, ElectricBarometer only supports training_mode='selection_only'."
            )

        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be strictly positive.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                "selection_mode must be either 'holdout' or 'cv'; "
                f"got {selection_mode!r}."
            )

        if selection_mode == "cv" and (cv is None or cv < 2):
            raise ValueError(
                f"In CV mode, cv must be at least 2; got {cv!r}."
            )

        self.models: Dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)
        self.selection_mode: str = selection_mode
        self.cv: int = int(cv)
        self.random_state: Optional[int] = random_state

        # Fitted state
        self.best_name_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.results_: Any = None  # pandas.DataFrame, but we avoid importing in signature

        # Validation metrics for the winning model
        self.validation_cwsl_: Optional[float] = None
        self.validation_rmse_: Optional[float] = None
        self.validation_wmape_: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def r_(self) -> float:
        """Return the cost ratio R = cu / co."""
        return self.cu / self.co

    # ------------------------------------------------------------------
    # Internal: K-fold CV evaluation
    # ------------------------------------------------------------------
    def _cv_evaluate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Perform simple K-fold CV over the candidate models and return
        a DataFrame with mean scores per model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        sample_weight : ndarray of shape (n_samples,), optional
            Optional per-sample weights. If provided, they are subset
            to each fold's validation indices and passed to the metric
            functions.

        Returns
        -------
        results : DataFrame
            Index: model (model name)
            Columns: ["CWSL", "RMSE", "wMAPE"]
        """
        n_samples = X.shape[0]
        k = self.cv

        if k < 2 or k > n_samples:
            raise ValueError(
                f"Invalid number of folds cv={k} for n_samples={n_samples}."
            )

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        # Compute fold sizes (as evenly as possible)
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[: n_samples % k] += 1

        folds = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, val_idx))
            current = stop

        rows = []

        for model_name, base_model in self.models.items():
            cwsl_scores = []
            rmse_scores = []
            wmape_scores = []

            for train_idx, val_idx in folds:
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                model = _clone_model(base_model)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)

                sw_va = None
                if sample_weight is not None:
                    sw_va = np.asarray(sample_weight, dtype=float)[val_idx]

                # Compute metrics for this fold
                cwsl_scores.append(
                    cwsl(
                        y_true=y_va,
                        y_pred=y_pred,
                        cu=self.cu,
                        co=self.co,
                        sample_weight=sw_va,
                    )
                )
                rmse_scores.append(
                    rmse(
                        y_true=y_va,
                        y_pred=y_pred,
                        sample_weight=sw_va,
                    )
                )
                wmape_scores.append(
                    wmape(
                        y_true=y_va,
                        y_pred=y_pred,
                        sample_weight=sw_va,
                    )
                )

            rows.append(
                {
                    "model": model_name,
                    "CWSL": float(np.mean(cwsl_scores)),
                    "RMSE": float(np.mean(rmse_scores)),
                    "wMAPE": float(np.mean(wmape_scores)),
                }
            )

        results = pd.DataFrame(rows).set_index("model")
        return results

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        refit_on_full: Optional[bool] = None,
    ) -> "ElectricBarometer":
        """
        Fit all candidate models and select the best one using CWSL.

        Parameters
        ----------
        X_train : array-like of shape (n_samples_train, n_features)
            - In holdout mode: the training subset.
            - In cv mode: the full dataset over which CV will be run.

        y_train : array-like of shape (n_samples_train,)
            Targets corresponding to X_train.

        X_val : array-like of shape (n_samples_val, n_features)
            - In holdout mode: the validation subset.
            - In cv mode: ignored (you may pass X_train again or any placeholder).

        y_val : array-like of shape (n_samples_val,)
            - In holdout mode: validation targets.
            - In cv mode: ignored.

        sample_weight_train : array-like of shape (n_samples_train,), optional
            - In holdout mode: currently ignored (reserved for future use).
            - In cv mode: treated as sample weights for the full dataset and
              used in the metric calculations for each validation fold.

        sample_weight_val : array-like of shape (n_samples_val,), optional
            (Currently ignored in v0.3.x; reserved for future use.)

        refit_on_full : bool, optional
            In holdout mode, if provided, overrides the instance-level
            refit_on_full flag for this .fit() call only. If None, uses
            self.refit_on_full. Ignored in cv mode.

        Returns
        -------
        self : ElectricBarometer
            The fitted selector, with best_model_ and results_ populated.
        """
        # Decide whether to refit on full data for this call (holdout mode only)
        refit_flag = self.refit_on_full if refit_on_full is None else bool(refit_on_full)

        if self.selection_mode == "holdout":
            # NOTE: select_model_by_cwsl currently does NOT accept sample_weight args,
            # so we ignore sample_weight_train/sample_weight_val here in v0.3.x.
            best_name, best_model, results = select_model_by_cwsl(
                models=self.models,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cu=self.cu,
                co=self.co,
            )

            self.best_name_ = best_name
            self.results_ = results

            # Extract validation metrics for the winner, if available
            self.validation_cwsl_ = None
            self.validation_rmse_ = None
            self.validation_wmape_ = None

            try:
                # results is expected to be a DataFrame with index as model names
                row = results.loc[best_name]
                if "CWSL" in row:
                    self.validation_cwsl_ = float(row["CWSL"])
                if "RMSE" in row:
                    self.validation_rmse_ = float(row["RMSE"])
                if "wMAPE" in row:
                    self.validation_wmape_ = float(row["wMAPE"])
            except Exception:
                # Be defensive: if results is not in the expected shape, just leave
                # the validation_* attributes as None.
                pass

            # Optionally refit the winning model on all available data
            best_model_refit = best_model
            if refit_flag and hasattr(best_model_refit, "fit"):
                X_full = np.concatenate([X_train, X_val], axis=0)
                y_full = np.concatenate([y_train, y_val], axis=0)

                best_model_refit = _clone_model(best_model_refit)
                best_model_refit.fit(X_full, y_full)

            self.best_model_ = best_model_refit

        else:  # selection_mode == "cv"
            # Run K-fold CV on X_train / y_train only (X_val, y_val ignored)
            results = self._cv_evaluate_models(
                X=X_train,
                y=y_train,
                sample_weight=sample_weight_train,
            )
            self.results_ = results

            # Pick the winner by lowest mean CWSL
            best_name = results["CWSL"].idxmin()
            self.best_name_ = best_name

            # Populate validation_* from mean CV scores
            row = results.loc[best_name]
            self.validation_cwsl_ = float(row["CWSL"])
            self.validation_rmse_ = float(row["RMSE"])
            self.validation_wmape_ = float(row["wMAPE"])

            # Fit a fresh clone of the winning model on **all** X_train / y_train
            base_model = self.models[best_name]
            best_model_refit = _clone_model(base_model)
            best_model_refit.fit(X_train, y_train)
            self.best_model_ = best_model_refit

        return self

    # ------------------------------------------------------------------
    # Prediction + scoring helpers
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from the selected best model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.best_model_ is None:
            raise RuntimeError(
                "ElectricBarometer has not been fit yet. "
                "Call .fit(...) first (holdout or cv mode)."
            )

        y_pred = self.best_model_.predict(X)
        return np.asarray(y_pred, dtype=float)

    def cwsl_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        cu: Optional[float] = None,
        co: Optional[float] = None,
    ) -> float:
        """
        Compute CWSL with this selector's cu/co (or overrides).

        Parameters
        ----------
        y_true : array-like
            Actual demand.

        y_pred : array-like
            Forecasted demand.

        sample_weight : array-like, optional
            Optional non-negative weights per interval.

        cu : float, optional
            Override for underbuild cost per unit. If None, uses self.cu.

        co : float, optional
            Override for overbuild cost per unit. If None, uses self.co.

        Returns
        -------
        float
            CWSL value for the given series.
        """
        cu_eff = float(self.cu if cu is None else cu)
        co_eff = float(self.co if co is None else co)

        return float(
            cwsl(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_eff,
                co=co_eff,
                sample_weight=sample_weight,
            )
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        model_names = list(self.models.keys())
        best = self.best_name_ if self.best_name_ is not None else "None"
        return (
            f"ElectricBarometer(models={model_names}, "
            f"cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"refit_on_full={self.refit_on_full}, "
            f"selection_mode={self.selection_mode!r}, "
            f"cv={self.cv}, random_state={self.random_state!r}, "
            f"best_name_={best!r})"
        )