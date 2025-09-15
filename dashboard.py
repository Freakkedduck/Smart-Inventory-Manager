import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

from forecasting import SalesForecastingModel, SimpleForecastingModel
from inventory_optimization import InventoryOptimizer
from data_generator import SalesDataGenerator

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Inventory Dashboard",
    layout="wide",
    page_icon="📊"
)

# -----------------------------------------------------------------------------
# Custom CSS for Spotify Theme
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #1DB954;
    }
    .section-card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .stMetric {
        background: #1E1E1E;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    .stMetric label {
        color: #AAAAAA;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_data(size="small"):
    generator = SalesDataGenerator()
    return generator.generate_data() if size == "medium" else generator.generate_data(
        start_date="2023-01-01", end_date="2023-12-31")

@st.cache_resource
def train_forecaster(df):
    model = SimpleForecastingModel()
    model.fit(df)
    metrics = model.evaluate(df)
    return model, metrics

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Controls")

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
# Main Dashboard
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>Sales Forecasting & Inventory Dashboard</h1>", unsafe_allow_html=True)

# ------------------ Key Metrics ------------------
total_sales = filtered_df["sales"].sum()
avg_daily_sales = filtered_df["sales"].mean()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Avg Daily Sales", f"${avg_daily_sales:,.1f}")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Tabs ------------------
sales_tab, forecast_tab, inventory_tab = st.tabs([
    "Sales Analysis", "Forecasting", "Inventory Management"])

# -----------------------------------------------------------------------------
# Sales Tab
# -----------------------------------------------------------------------------
with sales_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Sales Analysis")

    col1, col2 = st.columns(2)
    with col1:
        daily_sales = filtered_df.groupby("date")["sales"].sum().reset_index()
        fig_trend = px.line(daily_sales, x="date", y="sales", title="Daily Sales Trend", template="plotly_dark")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        family_sales = filtered_df.groupby("family")["sales"].sum().reset_index()
        fig_family = px.bar(family_sales, x="family", y="sales", title="Sales by Product Family", template="plotly_dark")
        st.plotly_chart(fig_family, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Forecast Tab
# -----------------------------------------------------------------------------
with forecast_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Sales Forecasting")

    model, perf = train_forecaster(filtered_df)
    last_date = filtered_df["date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        "date": future_dates,
        "store_nbr": filtered_df["store_nbr"].iloc[0],
        "family": filtered_df["family"].iloc[0],
        "sales": 0,
        "onpromotion": False
    })
    forecast_df = model.predict(pd.concat([filtered_df, future_df]))
    forecast_df = forecast_df[forecast_df["date"].isin(future_dates)]

    fig_forecast = px.line(forecast_df, x="date", y="predictions", title="30-Day Forecast", template="plotly_dark")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Model Performance on Historical Data")
    st.write(perf)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Inventory Tab
# -----------------------------------------------------------------------------
with inventory_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Inventory Management")

    optimizer = InventoryOptimizer()
    inv_results = optimizer.optimize_inventory(filtered_df, service_level=service_level, lead_time_days=lead_time)

    st.subheader("Reorder Recommendations")
    st.dataframe(inv_results[[
        "store_nbr", "family", "current_inventory", "reorder_point",
        "order_quantity", "needs_reorder"]], use_container_width=True)

    st.subheader("Summary")
    summary = optimizer.get_optimization_summary()
    st.write(summary)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Built by Kunal | Built with Streamlit")
