# CWSL — Cost-Weighted Service Loss  
A Framework for Operationally-Aligned Forecast Evaluation

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

# 9. scikit-learn Integration

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

# 10. Why CWSL Matters

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

# 11. Domains

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

# 12. Project Status

### Delivered
- Core metric suite  
- DataFrame utilities  
- Sensitivity analysis  
- Model comparison  
- scikit-learn integration  
- Keras-compatible CWSL loss  
- ElectricBarometer unified selector  
- Pipeline support  
- Cross-validation selection  

### Planned
- Full documentation site  

---

# 13. Contact

**Kyle Corrie (Economistician)**  
Creator of CWSL and the Forecast Readiness Framework  

- Email: kcorrie@economistician.com  
- LinkedIn: https://www.linkedin.com/in/kcorrie/