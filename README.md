# Sales Forecasting & Inventory Optimization Project

## Overview
This project delivers an end-to-end workflow that:
1. Generates realistic retail sales data (or loads your own)
2. Trains a machine-learning model for sales forecasting
3. Calculates safety stock, reorder points, and order quantities
4. Presents everything in an interactive Streamlit dashboard

---

## [Dashboard Link](https://freakkedduck-smart-inventory-manager-dashboard-ckyi6k.streamlit.app/)
<img width="277" height="290" alt="image" src="https://github.com/user-attachments/assets/fa45e262-6b88-4c88-b047-ef596f9cdf63" />

```
## Quick Start (local)
```bash
# 1. Clone repo & install deps
python -m venv venv && source venv/bin/activate  # win: venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run dashboard.py
```
## Command-Line Usage
```bash
# Train model + evaluate
python main.py --mode forecast

# Run inventory optimization @95% service level, 7-day lead time
python main.py --mode inventory --service-level 0.95 --lead-time 7
```
Generated CSVs:
* `forecast_results.csv`  – predictions & errors  
* `inventory_optimization.csv` – reorder recs  
* `abc_analysis.csv` – ABC classes

---
## Key Technologies
* **Python 3.8+**
* **scikit-learn** – Gradient Boosting / Linear Regression
* **pandas / numpy** – data wrangling
* **scipy** – safety-stock math
* **Streamlit 1.28** – interactive BI dashboard
* **Plotly 5** – rich charts

---

## Model Highlights
* Lag, rolling, calendar, holiday, and promo features
* Rolling-origin cross-validation
* Typical performance on sample data:
  * MAE ≈ 5.8 units  
  * MAPE ≈ 13.8 %  
  * RMSE ≈ 7.2 units

## Inventory Highlights
* Normal-approximation safety stock (`z·σ·√LT`)
* Reorder point = LT demand + safety stock
* EOQ helper in `inventory_optimization.py`

---

> **Need help?** Create a GitHub Issue or drop me a DM on LinkedIn.
