import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import DATA_PROCESSED, OUTPUTS

st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales & Inventory Intelligence")
st.caption(
    "Demand forecasting and inventory optimization "
    "| Synthetic retail simulation"
)

sales_path = DATA_PROCESSED / "clean_sales.csv"
forecast_path = OUTPUTS / "next_7_day_forecast.csv"
inventory_path = OUTPUTS / "inventory_recommendations.csv"
holdout_path = OUTPUTS / "holdout_predictions.csv"

required_files = [sales_path, forecast_path, inventory_path, holdout_path]
missing = [str(p) for p in required_files if not p.exists()]
if missing:
    st.error(
        "Required output files are missing. Run `python main.py` first. "
        f"Missing: {', '.join(missing)}"
    )
    st.stop()

sales = pd.read_csv(sales_path, parse_dates=["date"])
forecast = pd.read_csv(forecast_path, parse_dates=["date"])
inventory = pd.read_csv(inventory_path)
holdout = pd.read_csv(holdout_path, parse_dates=["date"])

st.sidebar.header("Dashboard Filters")
stores = sorted(sales["store_id"].dropna().unique().tolist())
products = sorted(sales["product_id"].dropna().unique().tolist())
categories = sorted(sales["category"].dropna().unique().tolist())

selected_stores = st.sidebar.multiselect("Select stores", stores, default=stores)
selected_products = st.sidebar.multiselect("Select products", products, default=products)
selected_categories = st.sidebar.multiselect(
    "Select categories", categories, default=categories
)

def apply_filters(frame):
    return frame[
        frame["store_id"].isin(selected_stores)
        & frame["product_id"].isin(selected_products)
        & frame["category"].isin(selected_categories)
    ].copy()

filtered_sales = apply_filters(sales)
filtered_forecast = apply_filters(forecast)
filtered_inventory = apply_filters(inventory)
filtered_holdout = apply_filters(holdout)

if filtered_sales.empty:
    st.warning("No historical sales match these filters. Change your selections.")
    st.stop()

# KPI cards always derive from the filtered data.
total_revenue = filtered_sales["revenue"].sum()
total_units = filtered_sales["units_sold"].sum()
product_count = filtered_sales["product_id"].nunique()
store_count = filtered_sales["store_id"].nunique()
reorder_count = int(filtered_inventory["reorder_required"].sum())
stockout_count = int(filtered_inventory["stockout_risk"].sum())
excess_count = int(filtered_inventory["excess_inventory"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Historical Revenue", f"{total_revenue:,.0f}")
k2.metric("Units Sold", f"{total_units:,.0f}")
k3.metric("Products", product_count)
k4.metric("Stores", store_count)
k5.metric("Reorder Alerts", reorder_count)

k6, k7, k8 = st.columns(3)
k6.metric("Stockout Risk", stockout_count)
k7.metric("Excess Stock Items", excess_count)
avg_cover = (
    filtered_inventory["days_of_inventory"].mean()
    if not filtered_inventory.empty else np.nan
)
k8.metric("Average Days of Inventory", f"{avg_cover:.1f}" if pd.notna(avg_cover) else "N/A")

st.divider()

st.subheader("Historical Revenue Trend")
daily_revenue = (
    filtered_sales.groupby("date", as_index=False)["revenue"].sum()
)
fig_revenue = px.line(
    daily_revenue, x="date", y="revenue",
    title="Daily Revenue", template="plotly_dark"
)
fig_revenue.update_layout(xaxis_title="Date", yaxis_title="Revenue")
st.plotly_chart(fig_revenue, use_container_width=True)

st.subheader("Forecast vs Actual — Holdout Evaluation")
if not filtered_holdout.empty:
    comparison = (
        filtered_holdout.groupby("date", as_index=False)[
            ["units_sold", "predicted_units", "seasonal_naive_units"]
        ].sum()
        .melt(
            id_vars="date",
            var_name="series",
            value_name="units"
        )
    )
    comparison["series"] = comparison["series"].map({
        "units_sold": "Actual sales",
        "predicted_units": "Random Forest",
        "seasonal_naive_units": "Seasonal naive"
    })
    fig_eval = px.line(
        comparison, x="date", y="units", color="series",
        title="Daily holdout demand: actual vs model vs baseline",
        template="plotly_dark"
    )
    fig_eval.update_layout(xaxis_title="Date", yaxis_title="Units sold")
    st.plotly_chart(fig_eval, use_container_width=True)

    rf_mae = (filtered_holdout["units_sold"] - filtered_holdout["predicted_units"]).abs().mean()
    naive_mae = (
        filtered_holdout["units_sold"] - filtered_holdout["seasonal_naive_units"]
    ).abs().mean()
    rf_rmse = np.sqrt(
        ((filtered_holdout["units_sold"] - filtered_holdout["predicted_units"]) ** 2).mean()
    )
    naive_rmse = np.sqrt(
        ((filtered_holdout["units_sold"] - filtered_holdout["seasonal_naive_units"]) ** 2).mean()
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest MAE", f"{rf_mae:.2f}")
    m2.metric("Seasonal Naive MAE", f"{naive_mae:.2f}")
    m3.metric("Random Forest RMSE", f"{rf_rmse:.2f}")
    m4.metric("Seasonal Naive RMSE", f"{naive_rmse:.2f}")
    if rf_mae < naive_mae:
        st.success(f"Random Forest MAE is {(1-rf_mae/naive_mae)*100:.1f}% lower than the seasonal-naive baseline for this filtered holdout.")
    else:
        st.warning("The seasonal-naive baseline performs as well as or better than Random Forest for this filtered holdout.")
else:
    st.info("No holdout records match the selected filters.")

st.subheader("Next 7-Day Demand Forecast")
if not filtered_forecast.empty:
    daily_forecast = (
        filtered_forecast.groupby("date", as_index=False)["forecast_units"].sum()
    )
    fig_forecast = px.line(
        daily_forecast, x="date", y="forecast_units",
        markers=True, title="Expected Daily Demand",
        template="plotly_dark"
    )
    fig_forecast.update_layout(xaxis_title="Forecast Date", yaxis_title="Forecast Units")
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("No forecast records match the selected filters.")

st.subheader("Inventory Risk & Replenishment")
if not filtered_inventory.empty:
    risk_filter = st.selectbox(
        "Inventory view",
        ["All products", "Reorder required", "Stockout risk", "Excess inventory"]
    )
    inventory_view = filtered_inventory.copy()
    if risk_filter == "Reorder required":
        inventory_view = inventory_view[inventory_view["reorder_required"]]
    elif risk_filter == "Stockout risk":
        inventory_view = inventory_view[inventory_view["stockout_risk"]]
    elif risk_filter == "Excess inventory":
        inventory_view = inventory_view[inventory_view["excess_inventory"]]

    display_columns = [
        "store_id", "product_id", "category",
        "on_hand_units_simulated", "forecast_daily_demand",
        "days_of_inventory", "lead_time_days",
        "reorder_point_units", "reorder_required",
        "recommended_order_units", "stockout_risk",
        "excess_inventory_units"
    ]
    st.dataframe(
        inventory_view[display_columns],
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        "`recommended_order_units` is zero unless `reorder_required` is True. "
        "Stockout risk means inventory position is below forecast demand during lead time. "
        "Excess inventory means stock exceeds the order-up-to target."
    )
else:
    st.info("No inventory records match the selected filters.")

st.subheader("Revenue by Category")
category_revenue = (
    filtered_sales.groupby("category", as_index=False)["revenue"]
    .sum()
    .sort_values("revenue", ascending=False)
)
fig_category = px.bar(
    category_revenue, x="category", y="revenue",
    color="category", title="Category-wise Revenue",
    template="plotly_dark"
)
st.plotly_chart(fig_category, use_container_width=True)

st.subheader("Business Insights")
if not category_revenue.empty:
    st.write(f"**Top revenue category:** {category_revenue.iloc[0]['category']}")
st.write(f"**Products requiring reorder:** {reorder_count}")
st.write(f"**Products with stockout risk:** {stockout_count}")
st.write(f"**Products with excess inventory:** {excess_count}")

st.info(
    "All inventory quantities are simulated for demonstration. "
    "This prototype is not connected to a live retail inventory or purchase-order system."
)
