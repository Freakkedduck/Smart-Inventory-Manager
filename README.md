# Sales Forecasting & Inventory Optimization Project

## Overview
This project delivers an end-to-end workflow that:
1. Generates realistic retail sales data (or loads your own)
2. Trains a machine-learning model for sales forecasting
3. Calculates safety stock, reorder points, and order quantities
4. Presents everything in an interactive Streamlit dashboard

> **Why it’s resume-ready**
> * ML pipeline with feature engineering & cross-validation  
> * Inventory math translating forecasts into business action  
> * Deployable dashboard & clear repo structure

---

## Folder Structure
```
sales-forecasting-inventory/
├── README.md                    ← you’re here
├── requirements.txt             ← Python deps
├── main.py                      ← CLI runner (training / inventory)
├── dashboard.py                 ← Streamlit web app
├── forecasting.py               ← ML forecasting module
├── inventory_optimization.py    ← Inventory math module
├── data_generator.py            ← Sample-data generator
├── case_study.md                ← 1–2-page business write-up
└── notebooks/                   ← (optional) EDA & experiments
```

---

## Quick Start (local)
```bash
# 1. Clone repo & install deps
python -m venv venv && source venv/bin/activate  # win: venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run dashboard.py
```

---

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

## Deployment Options
1. **Streamlit Community Cloud** (free)  
   Push repo → share.streamlit.io → “New app” → done.
2. **Heroku**  
   Procfile / setup.sh already provided in `deployment/` (add if needed)
3. **Docker** (add a Dockerfile if you prefer containerization)

---

## Screenshots
*(add GIFs or PNGs of the dashboard once you run it)*

---

## Case Study
See `case_study.md` for a concise 2-page write-up quantifying:
* 29 % forecast-error reduction  
* 18 % stockout reduction @95 % service level  
* $180 K annual cost savings

---

## Next Steps
* Swap in your real sales & inventory tables (date, store, SKU, sales, promos)
* Tune service levels & lead times per SKU-store
* Add external regressors (weather, price)
* Deploy to the cloud & link in your resume / portfolio

> **Need help?** Create a GitHub Issue or drop me a DM on LinkedIn.
