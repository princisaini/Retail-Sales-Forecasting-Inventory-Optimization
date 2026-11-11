
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import DATA_PROCESSED, OUTPUTS


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# Compact styling for a single-screen dashboard
st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    h1 {
        font-size: 1.65rem !important;
        padding-bottom: 0.3rem !important;
        margin-bottom: 0rem !important;
    }

    h2, h3 {
        font-size: 1.05rem !important;
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stMetric"] {
        padding: 0.35rem 0.5rem;
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    div[data-testid="stPlotlyChart"] {
        margin-bottom: -0.5rem;
    }

    hr {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }

    section[data-testid="stSidebar"] {
        min-width: 220px;
        max-width: 240px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Retail Sales & Inventory Intelligence")
st.caption(
    "Demand forecasting | Inventory optimization | Synthetic retail simulation"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

sales_path = DATA_PROCESSED / "clean_sales.csv"
forecast_path = OUTPUTS / "next_7_day_forecast.csv"
inventory_path = OUTPUTS / "inventory_recommendations.csv"
holdout_path = OUTPUTS / "holdout_predictions.csv"

required_files = [
    sales_path,
    forecast_path,
    inventory_path,
    holdout_path
]

missing = [str(p) for p in required_files if not p.exists()]

if missing:
    st.error(
        "Required files are missing. Run `python main.py` first. "
        + ", ".join(missing)
    )
    st.stop()

sales = pd.read_csv(sales_path, parse_dates=["date"])
forecast = pd.read_csv(forecast_path, parse_dates=["date"])
inventory = pd.read_csv(inventory_path)
holdout = pd.read_csv(holdout_path, parse_dates=["date"])


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

stores = sorted(sales["store_id"].dropna().unique().tolist())
products = sorted(sales["product_id"].dropna().unique().tolist())
categories = sorted(sales["category"].dropna().unique().tolist())

selected_stores = st.sidebar.multiselect(
    "Stores", stores, default=stores
)

selected_products = st.sidebar.multiselect(
    "Products", products, default=products
)

selected_categories = st.sidebar.multiselect(
    "Categories", categories, default=categories
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
    st.warning("No records match the selected filters.")
    st.stop()


# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

total_revenue = filtered_sales["revenue"].sum()
total_units = filtered_sales["units_sold"].sum()

product_count = filtered_sales["product_id"].nunique()
store_count = filtered_sales["store_id"].nunique()

reorder_count = int(filtered_inventory["reorder_required"].sum())
stockout_count = int(filtered_inventory["stockout_risk"].sum())
excess_count = int(filtered_inventory["excess_inventory"].sum())

avg_cover = (
    filtered_inventory["days_of_inventory"].mean()
    if not filtered_inventory.empty else np.nan
)


# --------------------------------------------------
# TOP KPI ROW
# --------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

k1.metric("Historical Revenue", f"{total_revenue:,.0f}")
k2.metric("Units Sold", f"{total_units:,.0f}")
k3.metric("Reorder Alerts", reorder_count)
k4.metric("Stockout Risks", stockout_count)

st.caption(
    f"Products: {product_count}  |  "
    f"Stores: {store_count}  |  "
    f"Excess stock items: {excess_count}  |  "
    f"Average inventory cover: "
    f"{avg_cover:.1f} days" if pd.notna(avg_cover)
    else "Inventory cover: N/A"
)

st.divider()


# --------------------------------------------------
# CHART 1: HISTORICAL REVENUE
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Revenue")

    daily_revenue = (
        filtered_sales.groupby("date", as_index=False)["revenue"].sum()
    )

    fig_revenue = px.line(
        daily_revenue,
        x="date",
        y="revenue",
        template="plotly_dark"
    )

    fig_revenue.update_layout(
        height=195,
        margin=dict(l=35, r=10, t=10, b=30),
        xaxis_title=None,
        yaxis_title="Revenue",
        showlegend=False
    )

    st.plotly_chart(
        fig_revenue,
        width="stretch",
        config={"displayModeBar": False},
        key="revenue_chart"
    )


# --------------------------------------------------
# CHART 2: NEXT 7-DAY FORECAST
# --------------------------------------------------

with col2:
    st.subheader("7-Day Demand Forecast")

    if not filtered_forecast.empty:

        daily_forecast = (
            filtered_forecast.groupby(
                "date", as_index=False
            )["forecast_units"].sum()
        )

        fig_forecast = px.line(
            daily_forecast,
            x="date",
            y="forecast_units",
            markers=True,
            template="plotly_dark"
        )

        fig_forecast.update_layout(
            height=195,
            margin=dict(l=35, r=10, t=10, b=30),
            xaxis_title=None,
            yaxis_title="Forecast Units",
            showlegend=False
        )

        st.plotly_chart(
            fig_forecast,
            width="stretch",
            config={"displayModeBar": False},
            key="forecast_chart"
        )

    else:
        st.info("No forecast records for these filters.")


# --------------------------------------------------
# CHART 3: HOLDOUT MODEL EVALUATION
# --------------------------------------------------

col3, col4 = st.columns(2)

with col3:
    st.subheader("Forecast vs Actual")

    if not filtered_holdout.empty:

        comparison = (
            filtered_holdout.groupby("date", as_index=False)[
                [
                    "units_sold",
                    "predicted_units",
                    "seasonal_naive_units"
                ]
            ].sum()
            .melt(
                id_vars="date",
                var_name="series",
                value_name="units"
            )
        )

        comparison["series"] = comparison["series"].map({
            "units_sold": "Actual",
            "predicted_units": "Random Forest",
            "seasonal_naive_units": "Seasonal Naive"
        })

        fig_eval = px.line(
            comparison,
            x="date",
            y="units",
            color="series",
            template="plotly_dark"
        )

        fig_eval.update_layout(
            height=195,
            margin=dict(l=35, r=10, t=10, b=30),
            xaxis_title=None,
            yaxis_title="Units",
            legend=dict(
                orientation="h",
                y=1.15,
                x=0
            )
        )

        st.plotly_chart(
            fig_eval,
            width="stretch",
            config={"displayModeBar": False},
            key="evaluation_chart"
        )

        actual = filtered_holdout["units_sold"]
        predicted = filtered_holdout["predicted_units"]
        naive = filtered_holdout["seasonal_naive_units"]

        rf_mae = (actual - predicted).abs().mean()
        naive_mae = (actual - naive).abs().mean()

        rf_rmse = np.sqrt(((actual - predicted) ** 2).mean())
        naive_rmse = np.sqrt(((actual - naive) ** 2).mean())

        st.caption(
            f"RF MAE: {rf_mae:.2f}  |  "
            f"Naive MAE: {naive_mae:.2f}  |  "
            f"RF RMSE: {rf_rmse:.2f}  |  "
            f"Naive RMSE: {naive_rmse:.2f}"
        )

    else:
        st.info("No holdout records for these filters.")


# --------------------------------------------------
# CHART 4: CATEGORY REVENUE
# --------------------------------------------------

with col4:
    st.subheader("Revenue by Category")

    category_revenue = (
        filtered_sales.groupby(
            "category", as_index=False
        )["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    fig_category = px.bar(
        category_revenue,
        x="category",
        y="revenue",
        color="category",
        template="plotly_dark"
    )

    fig_category.update_layout(
        height=195,
        margin=dict(l=35, r=10, t=10, b=35),
        xaxis_title=None,
        yaxis_title="Revenue",
        showlegend=False
    )

    st.plotly_chart(
        fig_category,
        width="stretch",
        config={"displayModeBar": False},
        key="category_chart"
    )


st.divider()


# --------------------------------------------------
# INVENTORY TABLE
# --------------------------------------------------

inventory_col, insight_col = st.columns([3, 1])

with inventory_col:

    st.subheader("Inventory Risk & Replenishment")

    if not filtered_inventory.empty:

        risk_filter = st.selectbox(
            "Inventory view",
            [
                "All products",
                "Reorder required",
                "Stockout risk",
                "Excess inventory"
            ],
            label_visibility="collapsed"
        )

        inventory_view = filtered_inventory.copy()

        if risk_filter == "Reorder required":
            inventory_view = inventory_view[
                inventory_view["reorder_required"]
            ]

        elif risk_filter == "Stockout risk":
            inventory_view = inventory_view[
                inventory_view["stockout_risk"]
            ]

        elif risk_filter == "Excess inventory":
            inventory_view = inventory_view[
                inventory_view["excess_inventory"]
            ]

        display_columns = [
            "store_id",
            "product_id",
            "category",
            "on_hand_units_simulated",
            "forecast_daily_demand",
            "reorder_point_units",
            "recommended_order_units",
            "stockout_risk",
            "excess_inventory_units"
        ]

        st.dataframe(
            inventory_view[display_columns],
            width="stretch",
            height=155,
            hide_index=True
        )

with insight_col:

    st.subheader("Business Insights")

    if not category_revenue.empty:
        st.write(
            f"**Top category:** "
            f"{category_revenue.iloc[0]['category']}"
        )

    st.write(f"**Reorder items:** {reorder_count}")
    st.write(f"**Stockout risks:** {stockout_count}")
    st.write(f"**Excess items:** {excess_count}")

    st.caption(
        "Inventory quantities are simulated. "
        "No live inventory or purchase-order integration."
    )
