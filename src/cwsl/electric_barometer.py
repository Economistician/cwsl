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
          - plus reference metrics (MAE, RMSE, wMAPE, etc.)
      * Selects the validation winner by **minimizing CWSL**.
      * Exposes a clean .fit() / .predict() API and a results_ DataFrame.

    This is the Phase-4 “selection-only” wrapper: it does **not** yet change
    the internal training objective of the models. It just makes sure you’re
    picking winners according to the right asymmetric cost metric.

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
        Tolerance for HR@τ in downstream diagnostics (not used for selection
        in v0.2.x, but kept as a configuration parameter for future use).

    training_mode : {"selection_only"}, default "selection_only"
        Reserved for future extension. In v0.2.x, only "selection_only"
        is supported (models train with their own objective; CWSL is used
        only for validation-time selection).

    Attributes
    ----------
    best_name_ : str or None
        Name of the selected best model after .fit().

    best_model_ : Any or None
        The selected model object itself (fitted).

    results_ : pandas.DataFrame or None
        Comparison table returned by `select_model_by_cwsl`, including:
        model name, CWSL, NSL, MAE, RMSE, etc., for each candidate.

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
    ) -> None:
        if not models:
            raise ValueError("ElectricBarometer requires at least one candidate model.")

        if training_mode != "selection_only":
            raise ValueError(
                "In v0.2.x, ElectricBarometer only supports training_mode='selection_only'."
            )

        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be strictly positive.")

        self.models: Dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode

        # Fitted state
        self.best_name_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.results_: Any = None  # pandas.DataFrame, but we avoid importing pandas here

    @property
    def r_(self) -> float:
        """Return the cost ratio R = cu / co."""
        return self.cu / self.co

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
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
        sample_weight_val : array-like of shape (n_samples_val,), optional

        Returns
        -------
        self : ElectricBarometer
            The fitted selector, with best_model_ and results_ populated.
        """
        # NOTE: select_model_by_cwsl currently does NOT accept sample_weight args,
        # so we ignore sample_weight_train/sample_weight_val here in v0.2.x.
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
        self.best_model_ = best_model
        self.results_ = results
        return self

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

    def __repr__(self) -> str:
        model_names = list(self.models.keys())
        best = self.best_name_ if self.best_name_ is not None else "None"
        return (
            f"ElectricBarometer(models={model_names}, "
            f"cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"best_name_={best!r})"
        )