import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import MODELS, OUTPUTS
from src.features import add_features, FEATURES, GROUPS


def train_and_evaluate(df):
    featured = add_features(df).dropna(subset=FEATURES + ["units_sold"])
    cutoff = featured["date"].quantile(0.80)
    train = featured[featured["date"] <= cutoff]
    test = featured[featured["date"] > cutoff]

    if train.empty or test.empty:
        raise ValueError("Not enough history for chronological train/test split.")

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[FEATURES], train["units_sold"])
    pred = np.maximum(0, model.predict(test[FEATURES]))

    # Seasonal-naive baseline: use the observed sales from the same weekday
    # one week earlier. This is a one-step-ahead holdout comparison.
    naive_pred = np.maximum(0, test["lag_7"].to_numpy(dtype=float))
    actual = test["units_sold"].to_numpy(dtype=float)

    metrics = {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "Seasonal_Naive_MAE": float(mean_absolute_error(actual, naive_pred)),
        "Seasonal_Naive_RMSE": float(mean_squared_error(actual, naive_pred) ** 0.5),
        "MAE_Improvement_vs_Naive_Pct": float(
            (1 - mean_absolute_error(actual, pred) /
             mean_absolute_error(actual, naive_pred)) * 100
        ),
        "RMSE_Improvement_vs_Naive_Pct": float(
            (1 - (mean_squared_error(actual, pred) ** 0.5) /
             (mean_squared_error(actual, naive_pred) ** 0.5)) * 100
        ),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "evaluation_note": (
            "One-step-ahead chronological holdout. Lag features during the "
            "test period use observed prior sales; this is not recursive "
            "multi-step backtesting."
        )
    }

    result = test[
        ["date", "store_id", "product_id", "category", "units_sold"]
    ].copy()
    result["predicted_units"] = pred
    result["seasonal_naive_units"] = naive_pred
    result.to_csv(OUTPUTS / "holdout_predictions.csv", index=False)

    joblib.dump(model, MODELS / "demand_model.joblib")
    return model, metrics, result


def forecast_next_days(df, model, horizon=7):
    """Generate recursive daily forecasts for each store-product combination."""
    forecasts = []

    for (store, product), group in df.groupby(GROUPS):
        history = group.sort_values("date").copy()
        last = history.iloc[-1]
        sales = history["units_sold"].astype(float).tolist()

        for step in range(1, horizon + 1):
            date = history["date"].max() + pd.Timedelta(days=step)
            dow = date.day_of_week
            row = {
                "lag_1": sales[-1],
                "lag_7": sales[-7] if len(sales) >= 7 else sales[-1],
                "rolling_mean_7": float(np.mean(sales[-7:])),
                "rolling_mean_28": float(np.mean(sales[-28:])),
                "day_of_week": dow,
                "month": date.month,
                "day_of_year": date.day_of_year,
                "is_weekend": int(dow in [5, 6]),
                "promotion": 0,
                "unit_price": float(last["unit_price"])
            }

            prediction = max(
                0.0,
                float(model.predict(pd.DataFrame([row])[FEATURES])[0])
            )
            sales.append(prediction)

            forecasts.append({
                "date": date,
                "store_id": store,
                "product_id": product,
                "category": last["category"],
                "forecast_units": round(prediction, 2),
                "unit_price": float(last["unit_price"])
            })

    output = pd.DataFrame(forecasts)
    output.to_csv(OUTPUTS / "next_7_day_forecast.csv", index=False)
    return output
