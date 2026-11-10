import numpy as np
import pandas as pd
from src.config import OUTPUTS


def create_inventory_recommendations(
    df,
    forecast,
    lead_time_days=5,
    service_level_z=1.65,
    review_period_days=7
):
    """
    Illustrative periodic-review inventory policy.

    Reorder point = lead-time demand + safety stock.
    Order-up-to = demand over lead time + review period + safety stock.

    Inventory is simulated. No live stock or purchase-order feed is used.
    """
    recommendations = []

    for (store, product), group in df.groupby(["store_id", "product_id"]):
        group = group.sort_values("date")
        recent = group.tail(56)["units_sold"].astype(float)
        daily_mean = float(recent.mean())
        daily_std = float(recent.std(ddof=1) or 0)

        product_forecast = forecast[
            (forecast["store_id"] == store)
            & (forecast["product_id"] == product)
        ]
        forecast_daily = (
            float(product_forecast["forecast_units"].mean())
            if not product_forecast.empty else daily_mean
        )

        lead_time_demand = forecast_daily * lead_time_days
        safety_stock = service_level_z * daily_std * np.sqrt(lead_time_days)
        reorder_point = lead_time_demand + safety_stock
        order_up_to = (
            forecast_daily * (lead_time_days + review_period_days)
            + safety_stock
        )

        # Reproducible simulated on-hand stock: approximately eight days
        # of recent average demand, with a small product-specific offset.
        on_hand = max(0, int(round(daily_mean * 8 + (len(product) * 3))))
        inventory_position = on_hand

        reorder_required = inventory_position <= reorder_point

        # Only recommend a purchase when the reorder condition is met.
        recommended_order = (
            max(0, int(np.ceil(order_up_to - inventory_position)))
            if reorder_required else 0
        )

        # Risk flag: stock position is below expected demand during supplier
        # lead time, before adding safety stock.
        stockout_risk = inventory_position < lead_time_demand

        # Excess is stock above the policy's order-up-to target.
        excess_units = max(0, int(np.floor(inventory_position - order_up_to)))

        days_of_inventory = (
            inventory_position / forecast_daily if forecast_daily > 0 else np.nan
        )

        recommendations.append({
            "store_id": store,
            "product_id": product,
            "category": group["category"].iloc[-1],
            "on_hand_units_simulated": on_hand,
            "forecast_daily_demand": round(forecast_daily, 2),
            "lead_time_days": lead_time_days,
            "safety_stock_units": int(np.ceil(safety_stock)),
            "reorder_point_units": int(np.ceil(reorder_point)),
            "order_up_to_units": int(np.ceil(order_up_to)),
            "inventory_position_units": inventory_position,
            "days_of_inventory": round(days_of_inventory, 2)
                if pd.notna(days_of_inventory) else np.nan,
            "stockout_risk": bool(stockout_risk),
            "excess_inventory_units": excess_units,
            "excess_inventory": bool(excess_units > 0),
            "reorder_required": bool(reorder_required),
            "recommended_order_units": recommended_order
        })

    result = pd.DataFrame(recommendations)
    result.to_csv(OUTPUTS / "inventory_recommendations.csv", index=False)
    return result
