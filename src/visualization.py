import matplotlib.pyplot as plt
import pandas as pd
from src.config import IMAGES, OUTPUTS

def create_charts(df, holdout, forecast, inventory):
    IMAGES.mkdir(parents=True, exist_ok=True)
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    plt.figure(figsize=(11, 5))
    plt.plot(daily.date, daily.revenue)
    plt.title("Daily Retail Revenue")
    plt.xlabel("Date"); plt.ylabel("Revenue")
    plt.tight_layout(); plt.savefig(IMAGES / "sales_trend.png", dpi=150); plt.close()

    category = df.groupby("category")["revenue"].sum().sort_values()
    plt.figure(figsize=(9, 5))
    category.plot(kind="barh")
    plt.title("Revenue by Product Category")
    plt.xlabel("Revenue")
    plt.tight_layout(); plt.savefig(IMAGES / "category_revenue.png", dpi=150); plt.close()

    sample = holdout[(holdout.store_id == holdout.store_id.iloc[0]) &
                     (holdout.product_id == holdout.product_id.iloc[0])].sort_values("date")
    plt.figure(figsize=(11, 5))
    plt.plot(sample.date, sample.units_sold, label="Actual")
    plt.plot(sample.date, sample.predicted_units, label="Predicted")
    plt.legend(); plt.title("Holdout: Actual vs Predicted Demand")
    plt.tight_layout(); plt.savefig(IMAGES / "actual_vs_predicted.png", dpi=150); plt.close()

    f = forecast.groupby("date", as_index=False)["forecast_units"].sum()
    plt.figure(figsize=(9, 5))
    plt.plot(f.date, f.forecast_units, marker="o")
    plt.title("Next 7 Days: Total Forecast Units")
    plt.tight_layout(); plt.savefig(IMAGES / "seven_day_forecast.png", dpi=150); plt.close()

    inventory.sort_values("recommended_order_units", ascending=False).to_csv(
        OUTPUTS / "inventory_recommendations_sorted.csv", index=False
    )