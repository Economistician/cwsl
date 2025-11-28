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

## Quick Start

### **Example 1 — Basic Python Usage**

```python
from cwsl import cwsl

y_true = [10, 12, 8]
y_pred = [9, 15, 7]

cu = 2.0  # shortfall cost
co = 1.0  # overbuild cost

score = cwsl(y_true, y_pred, cu=cu, co=co)
print(score)
```

### **Example 2 — DataFrame Workflow**

```python
import pandas as pd
from cwsl import compute_cwsl_df

df = pd.DataFrame({
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

## Model Comparison Example

You can compare multiple forecast models on the same actuals using `compare_forecasts`.

```python
import numpy as np
from cwsl import compare_forecasts

y_true = np.array([10, 12, 8])

forecasts = {
    "under_model": np.array([9, 11, 7]),
    "over_model":  np.array([12, 14, 10]),
    "naive_model": np.array([10, 12, 8]),
}

cu = 2.0  # shortfall cost
co = 1.0  # overbuild cost

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

## Included Metrics

This library implements CWSL and its supporting diagnostics:

### **Core Metric**
- **CWSL** – Cost-Weighted Service Loss

### **Diagnostics**
- **NSL** – No-Shortfall Level  
- **HR@τ** – Hit Rate within Tolerance  
- **UD** – Underbuild Depth  
- **wMAPE** – Weighted Mean Absolute Percentage Error  
- **FRS** – Forecast Readiness Score (`NSL - CWSL`)  

These metrics together provide a multidimensional view of **operational readiness**.

---

## Why CWSL Matters

In real-world operations:

- **Shortfalls** cause lost transactions, slower service, queue buildup, recovery delays, and negative customer experience.  
- **Overbuilds** typically cause minor waste or brief excess capacity.

Despite this, most organizations rely on **symmetric error metrics** that treat ± error as interchangeable.

CWSL exposes readiness-related failures that symmetric metrics consistently hide:

- Shortfall clustering at peak periods  
- Asymmetric operational consequences  
- Deep misses that matter more than shallow overbuilds  
- Interval-level vulnerability that MAE/MAPE smooth over  

If operational reliability matters, CWSL is the metric that aligns with reality.

---

## Who Is CWSL For?

CWSL is applicable to any domain with **short-horizon operational decisions**, including:

- Quick-service restaurants (QSR) & foodservice  
- Retail replenishment & on-shelf availability  
- Workforce & capacity planning  
- Manufacturing & production scheduling  
- Logistics & last-mile delivery  
- Energy & short-term load forecasting  
- Inventory & supply chain systems  

If being “short” is worse than being “long,” CWSL applies.

---

## Project Status

This project is under active development.

### **Planned for v0.1.0**

- [X] Implement core metrics (CWSL, NSL, HR@τ, UD, wMAPE, FRS)  
- [X] Add `cwsl_from_df` for item–interval DataFrame workflows  
- [X] Publish on PyPI (`pip install cwsl`)  
- [X] Add example notebooks (QSR, retail, workforce planning)  
- [X] Add visualization tools for asymmetric penalties  
- [X] Add CWSL-based model comparison utilities  

---

### **Planned for v0.2.0**

- [ ] Add more DataFrame utilities (multi-item, multi-store)
- [ ] Add scikit-learn scorer wrappers
- [ ] Add plot_cwsl_breakdown() visualization tools 
- [ ] Add asymmetric cost sensitivity analysis utilities 
- [ ] Add model comparison suite (compare_models_cwsl)
- [ ] Add documentation site (MkDocs + GitHub Pages) 

---

## Connect

If you'd like to discuss forecasting, operations, analytics, or the CWSL framework:

- Email: kcorrie@economistician.com
- LinkedIn: https://www.linkedin.com/in/kcorrie/

Created by **Kyle Corrie** (Economistician)
Founder of the CWSL Metric and the Forecast Readiness Framework