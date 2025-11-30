from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .compare import select_model_by_cwsl
from .metrics import cwsl


class ElectricBarometer:
    """
    ElectricBarometer: cost-aware model selector built on CWSL.

    This is a high-level wrapper that:

      * Takes a dictionary of candidate forecast models (typically scikit-learn
        regressors with .fit() / .predict() methods).
      * Trains all candidates "as usual" on a training set (they use their own
        internal loss: MSE, MAE, etc.).
      * Evaluates them on a validation set using:
          - CWSL (with your cu/co),
          - plus reference metrics (RMSE, wMAPE, etc.).
      * Selects the validation winner by **minimizing CWSL**.
      * Optionally refits the winning model on all available data
        (train ∪ validation) before exposing it via .best_model_.
      * Exposes a clean .fit() / .predict() API and a results_ DataFrame.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of candidate models. Keys are model names, values are
        estimator objects with scikit-learn style API:

            model.fit(X_train, y_train, sample_weight=...)
            model.predict(X_val)

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
        If True, after selecting the best model by CWSL on the validation
        set, refit that winning model on the concatenated (train ∪ val)
        data before exposing it via .best_model_ and .predict().

    Attributes
    ----------
    best_name_ : str or None
        Name of the selected best model after .fit().

    best_model_ : Any or None
        The selected model object itself (fitted). If refit_on_full=True
        (either at init or in the .fit() override), this is the model
        trained on all available data.

    results_ : pandas.DataFrame or None
        Comparison table returned by `select_model_by_cwsl`, with one row
        per candidate model and columns including CWSL, RMSE, wMAPE, etc.

    validation_cwsl_ : float or None
        CWSL value of the winning model on the validation set.

    validation_rmse_ : float or None
        RMSE value of the winning model on the validation set (if available).

    validation_wmape_ : float or None
        wMAPE value of the winning model on the validation set (if available).

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
    ) -> None:
        if not models:
            raise ValueError("ElectricBarometer requires at least one candidate model.")

        if training_mode != "selection_only":
            raise ValueError(
                "In v0.3.x, ElectricBarometer only supports training_mode='selection_only'."
            )

        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be strictly positive.")

        self.models: Dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)

        # Fitted state
        self.best_name_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.results_: Any = None  # pandas.DataFrame, but we avoid importing pandas here

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
        y_train : array-like of shape (n_samples_train,)
        X_val : array-like of shape (n_samples_val, n_features)
        y_val : array-like of shape (n_samples_val,)
        sample_weight_train : array-like of shape (n_samples_train,), optional
            (Currently ignored in v0.3.x; reserved for future use.)
        sample_weight_val : array-like of shape (n_samples_val,), optional
            (Currently ignored in v0.3.x; reserved for future use.)
        refit_on_full : bool, optional
            If provided, overrides the instance-level refit_on_full flag for
            this .fit() call only. If None, uses self.refit_on_full.

        Returns
        -------
        self : ElectricBarometer
            The fitted selector, with best_model_ and results_ populated.
        """
        # Decide whether to refit on full data for this call
        refit_flag = self.refit_on_full if refit_on_full is None else bool(refit_on_full)

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

            # Try to use sklearn.clone if available; otherwise, fall back
            # to a naive re-instantiation via get_params.
            cloned = None
            try:
                from sklearn.base import clone  # type: ignore

                cloned = clone(best_model_refit)
            except Exception:
                cloned = None

            if cloned is not None:
                best_model_refit = cloned
            else:
                # Fallback: re-create using class + get_params (if present)
                if hasattr(best_model_refit, "get_params"):
                    params = best_model_refit.get_params()
                    best_model_refit = best_model_refit.__class__(**params)

            best_model_refit.fit(X_full, y_full)

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
                "Call .fit(X_train, y_train, X_val, y_val) first."
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
            f"best_name_={best!r})"
        )