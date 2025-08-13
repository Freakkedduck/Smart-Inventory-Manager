"""
Dashboard Application
=====================

Streamlit dashboard for interactive sales forecasting and inventory optimization
analysis. This script ties together the forecasting and inventory modules and
provides an intuitive web interface for exploration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

from forecasting import SalesForecastingModel, SimpleForecastingModel
from inventory_optimization import InventoryOptimizer, ABCAnalysis
from data_generator import SalesDataGenerator

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Inventory Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_data(size="small"):
    """Generate or load sample data using the data generator."""
    generator = SalesDataGenerator()
    return generator.generate_data() if size == "medium" else generator.generate_data(
        start_date="2023-01-01", end_date="2023-12-31")

@st.cache_resource
def train_forecaster(df):
    """Train forecasting model and return fitted model & performance."""
    model = SimpleForecastingModel()
    model.fit(df)
    metrics = model.evaluate(df)
    return model, metrics

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Controls")

# Data selection
sample_size = st.sidebar.selectbox("Dataset Size", ["small", "medium"], index=0)

df = load_sample_data(sample_size)

stores = st.sidebar.multiselect(
    "Select Stores", df["store_nbr"].unique(), df["store_nbr"].unique())

families = st.sidebar.multiselect(
    "Select Product Families", df["family"].unique(), df["family"].unique())

filtered_df = df[(df["store_nbr"].isin(stores)) & (df["family"].isin(families))]

st.sidebar.markdown("---")
service_level = st.sidebar.slider("Service Level (%)", 85, 99, 95) / 100
lead_time = st.sidebar.slider("Lead Time (days)", 1, 14, 7)

# -----------------------------------------------------------------------------
# Main dashboard
# -----------------------------------------------------------------------------
st.title("📈 Sales Forecasting & Inventory Optimization Dashboard")

# Key metrics
total_sales = filtered_df["sales"].sum()
avg_daily_sales = filtered_df["sales"].mean()

col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Avg Daily Sales", f"${avg_daily_sales:,.1f}")

st.markdown("---")

# Tabs
sales_tab, forecast_tab, inventory_tab = st.tabs([
    "📊 Sales Analysis", "🔮 Forecasting", "📦 Inventory Management"])

# -----------------------------------------------------------------------------
# Sales Tab
# -----------------------------------------------------------------------------
with sales_tab:
    st.header("Sales Analysis")
    col1, col2 = st.columns(2)

    # Daily sales trend
    with col1:
        daily_sales = filtered_df.groupby("date")["sales"].sum().reset_index()
        fig_trend = px.line(daily_sales, x="date", y="sales", title="Daily Sales Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Sales by product family
    with col2:
        family_sales = filtered_df.groupby("family")["sales"].sum().reset_index()
        fig_family = px.bar(family_sales, x="family", y="sales", title="Sales by Product Family")
        st.plotly_chart(fig_family, use_container_width=True)

# -----------------------------------------------------------------------------
# Forecast Tab
# -----------------------------------------------------------------------------
with forecast_tab:
    st.header("Sales Forecasting")
    model, perf = train_forecaster(filtered_df)

    # Forecast next 30 days for combined selection
    last_date = filtered_df["date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        "date": future_dates,
        "store_nbr": filtered_df["store_nbr"].iloc[0],  # Assume first selection
        "family": filtered_df["family"].iloc[0],
        "sales": 0,
        "onpromotion": False
    })
    forecast_df = model.predict(pd.concat([filtered_df, future_df]))
    forecast_df = forecast_df[forecast_df["date"].isin(future_dates)]

    # Plot forecast
    fig_forecast = px.line(forecast_df, x="date", y="predictions", title="30-Day Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Show performance
    st.subheader("Model Performance on Historical Data")
    st.write(perf)

# -----------------------------------------------------------------------------
# Inventory Tab
# -----------------------------------------------------------------------------
with inventory_tab:
    st.header("Inventory Management")

    optimizer = InventoryOptimizer()
    inv_results = optimizer.optimize_inventory(
        filtered_df, service_level=service_level, lead_time_days=lead_time)

    st.subheader("Reorder Recommendations")
    st.dataframe(inv_results[[
        "store_nbr", "family", "current_inventory", "reorder_point",
        "order_quantity", "needs_reorder"]], use_container_width=True)

    # Summary
    summary = optimizer.get_optimization_summary()
    st.markdown("### Summary")
    st.write(summary)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit  ·  © 2025")