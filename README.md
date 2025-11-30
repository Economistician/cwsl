# CWSL

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  
![status](https://img.shields.io/badge/status-active-brightgreen)

**Cost-Weighted Service Loss (CWSL)** is a forecast evaluation framework designed for environments where under-forecasting and over-forecasting do *not* have the same operational cost.

Traditional metrics like MAE, RMSE, and MAPE treat being **5 units short** the same as being **5 units long**.  
CWSL makes this asymmetry explicit and quantifies its impact at the operational interval level.

---

## What is CWSL?

CWSL is a **demand-normalized, directionally-aware** forecast error metric that applies **higher penalties for shortfalls** and **lower penalties for overbuilds**, reflecting their true operational costs.

It is designed for high-frequency operational decision-making where the *timing* and *direction* of forecast error matter as much as magnitude.

CWSL incorporates:

- **Asymmetric penalties** (`cu` for shortfall, `co` for overbuild)  
- **Interval-level evaluation** (5–30 minute windows)  
- **Demand normalization** for cross-store and cross-item comparability  
- **Additivity**, enabling aggregation across items, categories, or stores  

---

## Mathematical Definition

For each interval \(i\):

- Actual demand: \(y_i\)  
- Forecast: \(\hat{y}_i\)  
- Shortfall:  
  \[
  s_i = \max(0, y_i - \hat{y}_i)
  \]
- Overbuild:  
  \[
  o_i = \max(0, \hat{y}_i - y_i)
  \]
- Penalties:  
  - \(c_{u,i}\): cost per unit of shortfall  
  - \(c_{o,i}\): cost per unit of overbuild  

Interval cost:
\[
\text{cost}_i = c_{u,i}\, s_i + c_{o,i}\, o_i
\]

CWSL:
\[
\text{CWSL} =
\frac{\sum_i \text{cost}_i}{\sum_i y_i}
\]

When penalties are symmetric (`cu == co`), **CWSL reduces exactly to a demand-normalized absolute error**, equivalent to wMAPE.

---

## Simple Numeric Examples

### Shortfall case
- Actual: 100  
- Forecast: 90  
- Shortfall: 10  
- `cu = 3`, `co = 1`  

Cost:
```
cost = 3 * 10 = 30
CWSL = 30 / 100 = 0.30
```

### Overbuild case
- Actual: 100  
- Forecast: 110  
- Overbuild: 10  
- `cu = 3`, `co = 1`

Cost:
```
cost = 1 * 10 = 10
CWSL = 10 / 100 = 0.10
```

---

## Quick Start

### Example 1 — Basic Python Usage

```python
from cwsl import cwsl

y_true = [10, 12, 8]
y_pred = [9, 15, 7]

cu = 2.0  # shortfall cost
co = 1.0  # overbuild cost

score = cwsl(y_true, y_pred, cu=cu, co=co)
print(score)
```

---

### Example 2 — DataFrame Workflow

```python
import pandas as pd
from cwsl import compute_cwsl_df

df = pd.DataDataFrame({
    "item": ["burger", "burger", "fries"],
    "actual": [10, 12, 8],
    "forecast": [9, 15, 7],
})

results = compute_cwsl_df(
    df,
    actual_col="actual",
    forecast_col="forecast",
    cu=2.0,
    co=1.0,
    groupby_cols=["item"]   # optional
)

print(results)
```

---

### Example 3 — Grouped Evaluation

```python
from cwsl import evaluate_groups_df

group_summary = evaluate_groups_df(
    df,
    group_cols=["store_id", "item_id"],
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
    tau=2.0,
)

print(group_summary.head())
```

---

### Example 4 — Hierarchical panel (multi-store, multi-item)

You can generate a long-form “panel” table of CWSL metrics across multiple
levels (overall, by store, by item, etc.) using `evaluate_panel_df`.

```python
from cwsl import evaluate_panel_df

# df has columns: store_id, item_id, actual_qty, forecast_qty
levels = {
    "overall": [],
    "by_store": ["store_id"],
    "by_item": ["item_id"],
    "by_store_item": ["store_id", "item_id"],
}

panel = evaluate_panel_df(
    df=df,
    levels=levels,
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
    tau=2.0,
)

print(panel.head())
```

This returns a tidy DataFrame with columns like:
```text
level | store_id | item_id | metric       | value
------+----------+---------+-------------+-------
overall        …           cwsl          …
by_store   101   NaN       cwsl          …
by_item    NaN   BURGER    n_intervals   …
…
```
You can export this to CSV, plug it into a BI tool, or join it into a metrics mart.

---

## Model Comparison Example

```python
import numpy as np
from cwsl import compare_forecasts

y_true = np.array([10, 12, 8])

forecasts = {
    "under_model": np.array([9, 11, 7]),
    "over_model":  np.array([12, 14, 10]),
    "naive_model": np.array([10, 12, 8]),
}

cu = 2.0
co = 1.0

df = compare_forecasts(
    y_true=y_true,
    forecasts=forecasts,
    cu=cu,
    co=co,
    tau=2.0,
)

print(df)
```

---

### Entity-level cost ratios (advanced)

In many use cases, a **single global cost ratio** 

> R = cu / co

is enough. For more advanced users, CWSL also supports **entity-level** cost
ratios Rₑ (e.g., per item, SKU, or location).

You can estimate Rₑ from historical forecast performance using a simple
"balance" method:

```python
from cwsl import estimate_entity_R_from_balance

entity_costs = estimate_entity_R_from_balance(
    df=df,                 # panel with entity, actual, forecast
    entity_col="entity",   # e.g., item_id, sku, product, store
    y_true_col="actual",
    y_pred_col="forecast",
    ratios=(0.5, 1.0, 2.0, 3.0),  # candidate R values
    co=1.0,                # base overbuild cost
    sample_weight_col=None,
)

print(entity_costs.head())
```

This returns one row per entity with:

- `entity`
- `R` (chosen cost ratio)
- `cu (R * co)`
- `co`
- `under_cost`, `over_cost`, and their absolute difference `diff`

You can then merge `cu` / `co` back into your panel and use them in
grouped evaluation:

```python
from cwsl import evaluate_groups_df

df_with_costs = df.merge(
    entity_costs[["entity", "cu", "co"]],
    on="entity",
    how="left",
)

summary = evaluate_groups_df(
    df=df_with_costs,
    group_cols=["entity"],
    actual_col="actual",
    forecast_col="forecast",
    cu="cu",   # per-row shortfall cost
    co="co",   # per-row overbuild cost
    tau=2.0,
)

print(summary.head())
```

This pattern lets you:

- Start with a **global** R (simple)
- Upgrade to **entity-level** Rₑ (advanced)
- Without changing the core CWSL API.

---

## Model Selection with CWSL

Once you have multiple candidate models (random forests, gradient boosting,
ARIMA-style models with tabular features, etc.), you can use CWSL as the
selection criterion instead of RMSE or MAE.

The helper `select_model_by_cwsl`:

- Fits each model on `(X_train, y_train)` using its native loss.
- Predicts on a validation set `(X_val)`.
- Computes CWSL, RMSE, and wMAPE on the validation targets `y_val`.
- Returns the model with the **lowest CWSL** (i.e., lowest cost under your
  asymmetric penalties).

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from cwsl import select_model_by_cwsl

models = {
    "rf": RandomForestRegressor(random_state=0),
    "gbr": GradientBoostingRegressor(random_state=0),
}

best_name, best_model, results = select_model_by_cwsl(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    cu=2.0,   # shortfall cost
    co=1.0,   # overbuild cost
)

print(results)
print("Best model by CWSL:", best_name)
```

This keeps training completely standard (RMSE-style loss inside each model),
but uses **CWSL as the referee** when choosing which model is operationally
best under asymmetric costs.

---

---

## ElectricBarometer Enhancements (v0.3.x)

`ElectricBarometer` now includes two production-oriented capabilities that make
it a complete cost-aware model selection and finalization tool:

### 1. Validation Metrics (`validation_cwsl_`, `validation_rmse_`, `validation_wmape_`)

After calling `.fit()`, the selector stores the validation performance of the
winning model:

- `validation_cwsl_` – CWSL on the validation set  
- `validation_rmse_` – RMSE on the validation set  
- `validation_wmape_` – weighted MAPE on the validation set  

You can access them directly:

```python
print("Validation CWSL:", eb.validation_cwsl_)
print("Validation RMSE:", eb.validation_rmse_)
print("Validation wMAPE:", eb.validation_wmape_)
```

These mirror scikit-learn’s `best_score_` pattern and make post-selection
diagnostics effortless.

### 2. Optional Full Refit `(refit_on_full=True)`

In many real workflows, once you identify the best model using the validation
set, the final estimator should be trained on **all available data**.

Enable this by passing:

```python
eb.fit(
    X_train, y_train,
    X_val, y_val,
    refit_on_full=True,
)
```

Behavior:

1. All candidate models train on (X_train, y_train)
2. ElectricBarometer selects the winner by minimum validation CWSL
3. If refit_on_full=True, the winning estimator is retrained on:

```ini
X_full = concat(X_train, X_val)
y_full = concat(y_train, y_val)
```

This provides unbiased model selection and full-data finalization—ideal for
production deployment.

```python
models = {
    "linear": LinearRegression(),
    "rf": RandomForestRegressor(random_state=0),
}

eb = ElectricBarometer(models=models, cu=2.0, co=1.0)

eb.fit(
    X_train, y_train,
    X_val, y_val,
    refit_on_full=True,
)

print("Selected model:", eb.best_name_)
print("Validation CWSL:", eb.validation_cwsl_)
```

---

## Using CWSL as a scorer in scikit-learn

You can also plug CWSL directly into scikit-learn search utilities
(`GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, etc.) using the
cwsl_scorer helper.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from cwsl import cwsl_scorer

model = RandomForestRegressor(random_state=0)

# Build a scorer with your chosen cost ratio R = cu / co
scorer = cwsl_scorer(cu=2.0, co=1.0)

grid = GridSearchCV(
    estimator=model,
    param_grid={"n_estimators": [50, 100, 200]},
    scoring=scorer,
    cv=3,
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CWSL-based score:", grid.best_score_)
```

Important: to match scikit-learn conventions, the scorer returned by
cwsl_scorer returns the negative CWSL:

> score = -CWSL

So:

- Lower CWSL → higher score → better model under your asymmetric cost
structure.
- You still read and report CWSL itself using the positive values from
`cwsl(...)` or from the evaluation utilities.

---

## ElectricBarometer — Unified Cost-Aware Model Selection

`ElectricBarometer` is the unified, high-level model-selection engine built
on top of CWSL. Instead of comparing models manually or relying on symmetric
metrics like RMSE, ElectricBarometer automatically:

1. Fits multiple candidate models  
2. Predicts on a validation set  
3. Computes CWSL, RMSE, and wMAPE  
4. Selects the best model *based solely on operational cost*  
5. Provides a clean comparison table and a ready-to-use fitted winner  

It is the simplest path from “I have models” → “which one is best under my
asymmetric costs?”.

---

### Why ElectricBarometer Exists

Most models train using symmetric losses. But operational environments are
not symmetric:

- Under-forecasting (shortfalls) is costly  
- Overbuilds are usually cheaper  
- RMSE/MAPE hide this asymmetry entirely  

ElectricBarometer makes the **asymmetric cost** the first-class decision
criterion.

---

### Quick Example

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

import numpy as np
from cwsl import ElectricBarometer

# Synthetic data
rng = np.random.RandomState(0)
X = rng.randn(500, 3)
y = 5.0 * X[:, 0] + rng.randn(500) * 0.3
y = y - y.min() + 1.0   # ensure strictly positive for CWSL

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Candidate models
models = {
    "dummy": DummyRegressor(strategy="mean"),
    "linear": LinearRegression(),
    "rf": RandomForestRegressor(n_estimators=200, random_state=0),
}

# Cost ratio: shortfalls cost 2× more than overbuilds
eb = ElectricBarometer(models=models, cu=2.0, co=1.0)

# Fit and select the best model
eb.fit(X_train, y_train, X_val, y_val)

print("Best model:", eb.best_name_)
print(eb.results_)
y_pred = eb.predict(X_val)
```

ElectricBarometer selects the model with the **lowest CWSL**.

### API Overview

```python
ElectricBarometer(models, cu=2.0, co=1.0, tau=2.0)
```

Attributes after .fit():

- `best_name_` – name of the winning model
- `best_model_` – the fitted best estimator
- `results_` – pandas DataFrame (CWSL, RMSE, wMAPE)

Methods:

- `fit(X_train, y_train, X_val, y_val)`
- `predict(X)`
- `cwsl_score(y_true, y_pred)`

### Refitting the winner on all available data

In many real deployments, you want to:

1. Use a validation split to **choose** the best model by CWSL, then  
2. **Refit the winner** on all available data before going live.

ElectricBarometer supports this directly:

```python
from cwsl import ElectricBarometer

eb = ElectricBarometer(models=models, cu=2.0, co=1.0)

# Use (X_train, y_train) and (X_val, y_val) to select the best model by CWSL,
# then refit that winner on the concatenated dataset (train ∪ val).
eb.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full=True,
)

y_pred = eb.predict(X_test)
```

### When to Use ElectricBarometer

Use it when:

- You have multiple candidate models
- You care about asymmetric cost (shortfall > overbuild)
- You want the model with the lowest operational loss, not lowest RMSE

Typical domains:

- QSR production forecasting
- Retail availability / replenishment
- Workforce scheduling
- Manufacturing throughput
- Supply chain / logistics
- Energy load forecasting

ElectricBarometer is essentially a **cost-aware AutoML selector** built for
operational forecasting.

---

## Why Sensitivity Analysis Matters

One of the core strengths of CWSL is its ability to reveal how different forecast models behave
when the operational cost of *under-forecasting* increases.

Traditional symmetric metrics (RMSE, MAE, MAPE) treat all error as interchangeable.  
CWSL makes the asymmetry explicit — and sensitivity analysis shows how model behavior changes
as the shortfall penalty **cu** becomes larger relative to the overbuild penalty **co**.

### What the analysis shows

- **Under-lean models** (those that frequently shortfall) become dramatically more expensive  
  as the cost ratio \( R = cu / co \) increases.

- **Over-lean models** remain stable across all R values, because their risk profile avoids costly shortfalls.

- **Naïve or balanced models** often deteriorate rapidly under higher R values, revealing operational weakness
  that symmetric metrics fail to surface.

### Why this matters in practice

In real operational environments — QSR production, retail availability, logistics capacity,
manufacturing throughput, and similar systems — being “short” is usually far more costly than being “long.”

Sensitivity analysis makes it possible to answer questions such as:

- *“Which model is safest if the cost of being short is underestimated?”*  
- *“Does this model become unstable when cu increases?”*  
- *“Is this model robust across a realistic range of operational scenarios?”*

By evaluating models across a range of R values, CWSL provides a **cost-aware** and
**operationally aligned** understanding of forecast performance — something no symmetric metric can offer.

---

## Included Metrics

### Core Metric
- **CWSL** – Cost-Weighted Service Loss

### Diagnostics
- **NSL** – No-Shortfall Level  
- **HR@τ** – Hit Rate within Tolerance  
- **UD** – Underbuild Depth  
- **wMAPE** – Weighted Mean Absolute Percentage Error  
- **FRS** – Forecast Readiness Score (`NSL - CWSL`)  

These metrics provide a multidimensional view of **operational readiness**.

---

## Choosing `cu`, `co`, and the Cost Ratio `R = cu/co`

CWSL makes asymmetric costs explicit by requiring two parameters:

- **cu** – penalty for shortfall (being *under*)
- **co** – penalty for overbuild (being *over*)

The ratio **R = cu / co** is the real decision variable.  
It expresses *how many times worse* a shortfall is compared to an overbuild.

### Example Interpretation
- `R = 1` → symmetric costs (short = long)  
- `R = 2` → shortfalls are twice as costly as overbuilds  
- `R = 3` → shortfalls are three times worse  
- `R = 0.5` → overbuilds are more costly (rare but possible)

Most operational environments care primarily about under-forecasting, so **R ≥ 1** is typical.

---

## Ways to Choose R

CWSL supports multiple workflows depending on your maturity.

### **1. Manual Selection (Simple & Explicit)**

```python
cu = 2.0  # shortfall costs 2× more
co = 1.0
```

This is the most common and most transparent approach.

### **2. Cost-Balance Estimation (Data-Driven)**

Use `estimate_R_cost_balance` to derive an R that balances historical
shortfall and overbuild magnitudes:

```python
from cwsl import estimate_R_cost_balance

R = estimate_R_cost_balance(y_true, y_pred)
print(R)
```

This finds the **cost ratio that makes shortfall-cost ≈ overbuild-cost**
on the historical forecast errors, offering a simple, data-driven default.

---

## Cost Sensitivity Analysis

Because the “true” R is sometimes uncertain, it can be helpful to test
forecast performance across a range of plausible values.

```python
from cwsl import cwsl_sensitivity

sens = cwsl_sensitivity(
    y_true,
    y_pred,
    R_list=(0.5, 1.0, 2.0, 3.0),  # default sweep
    co=1.0,
)
print(sens)
```

This evaluates CWSL at multiple R values and returns a dictionary:

```yaml
{0.5: 0.08, 1.0: 0.12, 2.0: 0.18, 3.0: 0.22}
```

Use cases:
- **Model selection**: choose the model that performs best across the entire R-range
- **Stress testing**: ensure your chosen model is robust to cost uncertainty
- **Operational decision-making**: understand how sensitive performance is to cost assumptions

---

### Advanced: Entity-level R (per item / SKU / entity)

In many real systems, not every item has the same asymmetry.  
You may want a **global** cost ratio \(R = cu / co\) for most use cases, but allow
**entity-specific** Rₑ for high-impact items (e.g., core entree vs. napkins).

CWSL provides two helpers for this:

1. `estimate_entity_R_from_balance(...)`  
   – scans a grid of candidate cost ratios for each entity and finds the Rₑ
   where underbuild vs overbuild cost are closest in magnitude.

2. `evaluate_panel_with_entity_R(...)`  
   – takes a panel of entity–interval data and an entity-level R table, and
   computes the full CWSL metric suite using **per-entity cuₑ, coₑ**.

Example:

```python
import pandas as pd
from cwsl import (
    estimate_entity_R_from_balance,
    evaluate_panel_with_entity_R,
)

# panel of interval-level data
# columns: entity, t, actual_qty, forecast_qty
df_panel = pd.DataFrame({
    "entity": ["A", "A", "B", "B", "C", "C"],
    "t":      [1, 2, 1, 2, 1, 2],
    "actual_qty":   [10, 12, 15, 20,  8,  9],
    "forecast_qty": [11, 14, 14, 19,  7, 10],
})

# 1) Estimate Rₑ for each entity using a small grid and co = 1.0
entity_R = estimate_entity_R_from_balance(
    df=df_panel,
    entity_col="entity",
    y_true_col="actual_qty",
    y_pred_col="forecast_qty",
    ratios=(0.5, 1.0, 2.0, 3.0),
    co=1.0,
    sample_weight_col=None,
)

print(entity_R)
# columns: entity, R, cu, co, under_cost, over_cost, diff

# 2) Evaluate the panel using those entity-specific Rₑ values
summary = evaluate_panel_with_entity_R(
    df=df_panel,
    entity_R=entity_R,
    entity_col="entity",
    y_true_col="actual_qty",
    y_pred_col="forecast_qty",
    tau=2.0,
    sample_weight_col=None,
)

print(summary)
# one row per entity with:
#   entity, R, cu, co,
#   CWSL, NSL, UD, wMAPE, HR@tau,
#   FRS, MAE, RMSE, MAPE
```

This pattern lets you start with a **global R** for simplicity, then selectively
introduce **entity-level Rₑ** for items where asymmetry matters most.

---

## Why CWSL Matters

In real-world operations:

- **Shortfalls** cause lost transactions, slower service, queue buildup, and degraded customer experience.  
- **Overbuilds** typically cause minor waste or temporary excess capacity.  

Yet most organizations use symmetric metrics (MAE/MAPE/RMSE) that treat ± error as interchangeable.

CWSL exposes what these metrics hide:

- Shortfall clustering at peak periods  
- Asymmetric operational consequences  
- High-cost misses vs low-cost overshoots  
- Interval-level vulnerability  

If operational reliability matters, **CWSL matches reality**.

---

## Who Is CWSL For?

CWSL applies to any domain with **short-horizon operational decisions**, including:

- Quick-service restaurants (QSR)  
- Retail replenishment & on-shelf availability  
- Workforce & capacity planning  
- Manufacturing & production scheduling  
- Logistics & last-mile delivery  
- Energy & short-term load forecasting  
- Supply chain & inventory management  

If being “short” is worse than being “long,” CWSL is the right metric.

---

## Project Status

This project is under active development.

### Planned for v0.1.0
- [X] Core metrics implemented  
- [X] `cwsl_from_df` & DataFrame utilities  
- [X] Published on PyPI  
- [X] Example notebooks  
- [X] Visualization tools  
- [X] Model comparison utilities  

### Planned for v0.2.0
- [X] More DataFrame utilities  
- [X] scikit-learn wrappers  
- [X] `plot_cwsl_breakdown()`  
- [X] Cost sensitivity analysis  
- [X] CWSL model comparison suite  
- [X] Full documentation site  

### Planned / delivered in v0.3.0
- [X] ElectricBarometer unified selector  
- [X] scikit-learn wrappers (`cwsl_scorer`, selection helpers)  
- [X] Keras-compatible CWSL loss  
- [X] Cost sensitivity analysis  
- [ ] Full documentation site 

---

## Connect

If you'd like to discuss forecasting, operations, analytics, or the CWSL framework:

- **Email:** kcorrie@economistician.com  
- **LinkedIn:** https://www.linkedin.com/in/kcorrie/

Created by **Kyle Corrie (Economistician)**  
Founder of the CWSL Metric and the Forecast Readiness Framework