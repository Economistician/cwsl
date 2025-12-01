# CWSL — Cost-Weighted Service Loss  
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![status](https://img.shields.io/badge/status-active-brightgreen)

---

## Overview

**Cost-Weighted Service Loss (CWSL)** is a forecasting metric designed for environments where **shortfalls** and **overbuilds** do *not* have equal cost.

Traditional metrics (MAE, RMSE, MAPE) treat +5 and –5 errors identically.  
CWSL makes the asymmetry explicit, with:

- **cu** — cost of a shortfall  
- **co** — cost of an overbuild  
- **R = cu / co** — operational cost ratio  

CWSL is:

- **Asymmetric**
- **Demand-normalized**
- **Interval-level**
- **Additive and aggregable**
- **Operationally aligned**

It is built for real-world operational forecasting (QSR, retail, logistics, staffing, energy, manufacturing, supply-chain, etc.).

---

# 1. The CWSL Metric

## Mathematical Definition

For each interval \(i\):

\[
s_i = \max(0, y_i - \hat{y}_i)
\]
\[
o_i = \max(0, \hat{y}_i - y_i)
\]
\[
\text{cost}_i = c_{u,i}s_i + c_{o,i}o_i
\]

CWSL:

\[
\text{CWSL} = \frac{\sum_i \text{cost}_i}{\sum_i y_i}
\]

If `cu == co`, CWSL collapses exactly into **wMAPE**.

---

## Simple Numerical Example

### Shortfall
Actual 100, Forecast 90  
cu = 3, co = 1

```
shortfall = 10
cost = 3 * 10 = 30
CWSL = 30 / 100 = 0.30
```

### Overbuild
Actual 100, Forecast 110  
cu = 3, co = 1

```
overbuild = 10
cost = 1 * 10 = 10
CWSL = 10 / 100 = 0.10
```

Shortfalls hurt more — CWSL quantifies that.

---

# 2. Quick Start

## Basic Python Usage

```python
from cwsl import cwsl

y_true = [10, 12, 8]
y_pred = [9, 15, 7]

cwsl(y_true, y_pred, cu=2.0, co=1.0)
```

---

# 3. DataFrame Utilities

The library provides high-level helpers for grouped and hierarchical evaluation.

## Per-group evaluation

```python
from cwsl import evaluate_groups_df

summary = evaluate_groups_df(
    df,
    group_cols=["store_id", "item_id"],
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
)
```

## Multi-level panel evaluation

```python
from cwsl import evaluate_panel_df

panel = evaluate_panel_df(
    df=df,
    levels={
        "overall": [],
        "by_store": ["store_id"],
        "by_item": ["item_id"],
        "by_store_item": ["store_id", "item_id"],
    },
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
)
```

---

# 4. Model Comparison

## Compare multiple forecasts directly

```python
from cwsl import compare_forecasts

compare_forecasts(
    y_true,
    forecasts={"a": y_pred_a, "b": y_pred_b},
    cu=2.0,
    co=1.0,
)
```

Outputs a DataFrame with:

- CWSL  
- NSL  
- UD  
- wMAPE  
- HR@τ  
- FRS  
- MAE  
- RMSE  
- MAPE  

---

# 5. Choosing `cu`, `co`, and the Cost Ratio R

## Interpretation

- **R = 1** → symmetric costs  
- **R = 2** → shortfalls are twice as costly  
- **R = 3** → shortfalls are three times worse  
- **R < 1** → overbuilds cost more (rare)

Most operational use cases use **R ≥ 1**.

---

## Ways to choose R

### 1. Manual (simple)

```python
cu = 2.0
co = 1.0
```

### 2. Data-driven balance

```python
from cwsl import estimate_R_cost_balance
estimate_R_cost_balance(y_true, y_pred)
```

### 3. Entity-level R (advanced)

Estimate per-item, per-SKU, or per-store cost ratios:

```python
from cwsl import estimate_entity_R_from_balance
```

Then evaluate using:

```python
from cwsl import evaluate_panel_with_entity_R
```

---

# 6. Sensitivity Analysis

Forecast models behave differently as **cu / co** changes.  
CWSL provides sensitivity analysis:

```python
from cwsl import cwsl_sensitivity

cwsl_sensitivity(
    y_true,
    y_pred,
    R_list=(0.5, 1.0, 2.0, 3.0),
)
```

Outputs cost under each scenario — extremely useful for:

- stress-testing  
- model robustness  
- cost-aligned decision-making  

---

# 7. Included Metrics Suite

### Core
- **CWSL** — asymmetric cost penalty

### Diagnostics
- **NSL** — No-Shortfall Level  
- **UD** — Underbuild Depth  
- **HR@τ** — Hit Rate within tolerance  
- **wMAPE**  
- **FRS** — Forecast Readiness Score  
- **MAE**, **RMSE**, **MAPE**

---

# 8. ElectricBarometer — Unified Cost-Aware Model Selection

`ElectricBarometer` is the high-level model-selection engine.

It:

1. Accepts multiple candidate models (or pipelines)  
2. Trains them normally  
3. Evaluates using CWSL  
4. Selects the winner  
5. Optionally refits the winner on full data  
6. Supports both **holdout** and **cross-validation** selection  

---

## Example (holdout mode)

```python
from cwsl import ElectricBarometer
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

models = {
    "dummy": DummyRegressor(),
    "linear": LinearRegression(),
}

eb = ElectricBarometer(models=models, cu=2.0, co=1.0)

eb.fit(X_train, y_train, X_val, y_val)
print(eb.best_name_)
y_pred = eb.predict(X_val)
```

---

## Full refit (production-ready)

```python
eb.fit(
    X_train, y_train,
    X_val, y_val,
    refit_on_full=True,
)
```

Winner is retrained on **all** available data.

---

## Cross-validation selection

```python
eb = ElectricBarometer(
    models=models,
    cu=2.0, co=1.0,
    selection_mode="cv",
    cv=3,
    random_state=42,
)

eb.fit(X, y, X, y)
```

---

## Pipeline support

Fully compatible with scikit-learn Pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "linear": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
}
```

ElectricBarometer safely clones and evaluates each pipeline.

---

# 9. Model Engine Compatibility

ElectricBarometer works with **any engine** implementing:

```python
fit(X, y)
predict(X)
```

This includes:

**scikit-learn models**
- LinearRegression
- RandomForestRegressor
- GradientBoostingRegressor
- SVM
- Pipelines (scaler → model)

**XGBoost (sklearn API)**

```python
from xgboost import XGBRegressor
```

**LightGBM (sklearn API)**

```python
from lightgbm import LGBMRegressor
```

**statsmodels (via adapter)**

```python
class StatsmodelsAdapter:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fitted = self.model_class(y, **self.kwargs).fit()
        return self

    def predict(self, X):
        return self.fitted.forecast(len(X))
```

**Custom / proprietary engines**

```python
class MyCustomModel:
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
```

ElectricBarometer is a **universal cost-aware selector** — plug in anything that can fit + predict.

---

# 10. Model Engine Compatibility (Adapters)

ElectricBarometer includes built-in adapters for engines that do not follow `fit/predict` (Prophet, SARIMAX), and also supports modern gradient-boosting frameworks that *do* expose sklearn-style APIs:

- **XGBoost** (`XGBRegressor`)
- **LightGBM** (`LGBMRegressor`)
- **CatBoost** (`CatBoostRegressor`)

If a model can run inside a scikit-learn pipeline, it can run inside ElectricBarometer.

ElectricBarometer works with any model exposing:

```python
fit(X, y)
predict(X)
```

Most scikit-learn models already follow this pattern.
For engines that do **not** (Prophet, statsmodels, ARIMA, custom systems), CWSL provides lightweight *adapters* that wrap them with a sklearn-style API.

Built-in adapters

- **ProphetAdapter** -- wraps `prophet.Prophet`
- **SarimaxAdapter** -- wraps `statsmodels.tsa.statespace.SARIMAX`
- **AarimaAdapter** -- wraps `statsmodels.tsa.arima.model.ARIMA`

Example:

```python
from cwsl import ElectricBarometer, ProphetAdapter

models = {
    "baseline": DummyRegressor(strategy="mean"),
    "prophet": ProphetAdapter(),
}

eb = ElectricBarometer(models=models, cu=2.0, co=1.0)
eb.fit(X_train, y_train, X_val, y_val)
```

**Custom adapters**

You can adapt any engine with a small wrapper:

```python
class MyAdapter:
    def fit(self, X, y):
        ...
        return self
    def predict(self, X):
        ...
```

Adapters allow ElectricBarometer to be used with **any** forecasting engine while still selecting models using **CWSL**, **RMSE**, and **wMAPE**.

---

# 11. scikit-learn Integration

CWSL can be used directly as a scorer:

```python
from cwsl import cwsl_scorer
```

Use with GridSearchCV / RandomizedSearchCV:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=model,
    param_grid=...,
    scoring=cwsl_scorer(cu=2.0, co=1.0),
    cv=3,
)
```

`cwsl_scorer` returns **negative CWSL** to fit sklearn conventions.

---

# 12. CWSLRegressor -- Cost-Aware Ensemble Model Selector

`CWSLRegressor` is a scikit-learn–compatible estimator that evaluates multiple candidate models and automatically selects the one that minimizes **CWSL** on validation data.

It acts as a **drop-in forecasting model** that is cost-optimized and operationally aligned.

**Key Features**

- Accepts any models with `fit()` and `predict()`
- Uses **CWSL**, RMSE, and wMAPE for evaluation
- Supports **holdout** or **cross-validation** selection
- Optionally refits the winning model on the full dataset
- Fully sklearn-compatible (`fit/predict/get_params`)

**Basic Example**

```python
from cwsl import CWSLRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

models = {
    "linear": LinearRegression(),
    "rf": RandomForestRegressor(n_estimators=200, random_state=0),
}

reg = CWSLRegressor(
    models=models,
    cu=2.0,
    co=1.0,
    selection_mode="cv",
    cv=3,
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Selected model:", reg.best_name_)
```

**Why use CWSLRegressor?**

- Makes **model selection align with operational cost**, not generic RMSE
- Works seamlessly with traditional ML models, ensembles, pipelines, and adapters
- Reduces the risk of choosing a model that performs well statistically but poorly operationally

---

# 12. Why CWSL Matters

Operational asymmetry is real:

- Shortfalls → lost transactions, service failures, downtime  
- Overbuilds → mild waste or idle capacity  

Symmetric metrics hide this.  
CWSL exposes it.

CWSL reveals:

- peak-period vulnerability  
- clustering of shortfalls  
- asymmetric operational consequences  
- true cost of forecast error  

If being “short” is worse than being “long,” CWSL is the correct metric.

---

# 13. Domains

CWSL is used in:

- quick-service restaurants (QSR)  
- retail replenishment  
- workforce scheduling  
- manufacturing throughput  
- supply chain  
- logistics  
- energy load  
- short-term capacity planning  

---

# 14. Project Status

### Delivered
- Core metric suite  
- DataFrame utilities  
- Sensitivity analysis  
- Model comparison  
- scikit-learn integration  
- Keras-compatible CWSL loss  
- ElectricBarometer unified selector  
- Pipeline support  
- Adapters (Prophet, SARIMAX, ARIMA, LightGBM, CatBoost)
- Cross-validation selection  

### Planned
- Full documentation site  

---

# 15. Contact

**Kyle Corrie (Economistician)**  
Creator of CWSL and the Forecast Readiness Framework  

- Email: kcorrie@economistician.com  
- LinkedIn: https://www.linkedin.com/in/kcorrie/