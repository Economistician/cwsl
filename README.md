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
by_item    NaN   WHOPPER   n_intervals   …
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
print(R
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
- [ ] More DataFrame utilities  
- [ ] scikit-learn wrappers  
- [ ] `plot_cwsl_breakdown()`  
- [X] Cost sensitivity analysis  
- [X] CWSL model comparison suite  
- [ ] Full documentation site  

---

## Connect

If you'd like to discuss forecasting, operations, analytics, or the CWSL framework:

- **Email:** kcorrie@economistician.com  
- **LinkedIn:** https://www.linkedin.com/in/kcorrie/

Created by **Kyle Corrie (Economistician)**  
Founder of the CWSL Metric and the Forecast Readiness Framework