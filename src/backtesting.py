
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import DATA_PROCESSED, OUTPUTS
from src.features import add_features, FEATURES, GROUPS


HORIZON = 7
N_ORIGINS = 12
MIN_TRAIN_DAYS = 60


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def recursive_forecast(history, model, cutoff, horizon=HORIZON):
    """Forecast multiple days recursively using predictions, not future actuals."""
    history = history.sort_values("date").copy()

    sales = history["units_sold"].astype(float).tolist()
    last = history.iloc[-1]

    forecasts = []

    for step in range(1, horizon + 1):
        forecast_date = cutoff + pd.Timedelta(days=step)
        dow = forecast_date.day_of_week

        row = {
            "lag_1": sales[-1],
            "lag_7": sales[-7] if len(sales) >= 7 else sales[-1],
            "rolling_mean_7": float(np.mean(sales[-7:])),
            "rolling_mean_28": float(np.mean(sales[-28:])),
            "day_of_week": dow,
            "month": forecast_date.month,
            "day_of_year": forecast_date.day_of_year,
            "is_weekend": int(dow in [5, 6]),
            "promotion": 0,
            "unit_price": float(last["unit_price"]),
        }

        prediction = max(
            0.0,
            float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        )

        forecasts.append({
            "date": forecast_date,
            "horizon_day": step,
            "forecast_units": prediction,
        })

        # Feed the prediction into history for the next forecast day.
        sales.append(prediction)

    return forecasts


def run_rolling_backtest():
    data_path = DATA_PROCESSED / "clean_sales.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run python main.py first."
        )

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(GROUPS + ["date"]).reset_index(drop=True)

    required = GROUPS + [
        "date", "category", "unit_price", "promotion", "units_sold"
    ]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    first_date = dates.min()
    last_date = dates.max()

    earliest_cutoff = first_date + pd.Timedelta(days=MIN_TRAIN_DAYS)
    latest_cutoff = last_date - pd.Timedelta(days=HORIZON)

    eligible_dates = dates[
        (dates >= max(earliest_cutoff, last_date - pd.Timedelta(days=180)))
        & (dates <= latest_cutoff)
    ]

    if len(eligible_dates) == 0:
        raise ValueError("Not enough history for rolling-origin backtesting.")

    selected_positions = np.linspace(
        0,
        len(eligible_dates) - 1,
        num=min(N_ORIGINS, len(eligible_dates)),
        dtype=int,
    )
    cutoffs = eligible_dates[selected_positions].drop_duplicates()

    print(f"Dataset: {first_date.date()} to {last_date.date()}")
    print(f"Backtesting origins: {len(cutoffs)}")
    print(f"Forecast horizon: {HORIZON} days")

    actual_lookup = {
        (row.store_id, row.product_id, row.date): float(row.units_sold)
        for row in df.itertuples(index=False)
    }

    all_predictions = []

    for origin_number, cutoff in enumerate(cutoffs, start=1):
        print(f"\nOrigin {origin_number}/{len(cutoffs)}: {cutoff.date()}")

        train_raw = df[df["date"] <= cutoff].copy()

        train_features = add_features(train_raw)
        train_features = train_features.dropna(
            subset=FEATURES + ["units_sold"]
        )

        if train_features.empty:
            raise ValueError(f"No usable training rows at {cutoff.date()}.")

        model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train_features[FEATURES],
            train_features["units_sold"],
        )

        for (store, product), history in train_raw.groupby(GROUPS):
            history = history.sort_values("date").copy()

            if len(history) < 28:
                continue

            forecasts = recursive_forecast(
                history=history,
                model=model,
                cutoff=cutoff,
                horizon=HORIZON,
            )

            for item in forecasts:
                forecast_date = item["date"]
                actual_key = (store, product, forecast_date)

                if actual_key not in actual_lookup:
                    continue

                # Seasonal naive: same weekday from the previous week.
                naive_date = forecast_date - pd.Timedelta(days=7)
                naive_key = (store, product, naive_date)

                if naive_key not in actual_lookup:
                    continue

                all_predictions.append({
                    "cutoff_date": cutoff,
                    "date": forecast_date,
                    "horizon_day": item["horizon_day"],
                    "store_id": store,
                    "product_id": product,
                    "actual_units": actual_lookup[actual_key],
                    "predicted_units": item["forecast_units"],
                    "seasonal_naive_units": actual_lookup[naive_key],
                })

    predictions = pd.DataFrame(all_predictions)

    if predictions.empty:
        raise ValueError("No backtest predictions were generated.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        OUTPUTS / "rolling_backtest_predictions.csv",
        index=False,
    )

    summary_rows = []

    # Report accuracy separately for each forecast horizon.
    for horizon_day, group in predictions.groupby("horizon_day"):
        actual = group["actual_units"].to_numpy()
        predicted = group["predicted_units"].to_numpy()
        naive = group["seasonal_naive_units"].to_numpy()

        model_metrics = calculate_metrics(actual, predicted)
        naive_metrics = calculate_metrics(actual, naive)

        summary_rows.append({
            "forecast_horizon": f"Day {horizon_day}",
            "observations": len(group),
            "MAE": model_metrics["MAE"],
            "RMSE": model_metrics["RMSE"],
            "Seasonal_Naive_MAE": naive_metrics["MAE"],
            "Seasonal_Naive_RMSE": naive_metrics["RMSE"],
            "MAE_Improvement_Pct": (
                (1 - model_metrics["MAE"] / naive_metrics["MAE"]) * 100
                if naive_metrics["MAE"] > 0 else np.nan
            ),
            "RMSE_Improvement_Pct": (
                (1 - model_metrics["RMSE"] / naive_metrics["RMSE"]) * 100
                if naive_metrics["RMSE"] > 0 else np.nan
            ),
        })

    # Overall accuracy across all origins and all seven forecast days.
    actual = predictions["actual_units"].to_numpy()
    predicted = predictions["predicted_units"].to_numpy()
    naive = predictions["seasonal_naive_units"].to_numpy()

    model_metrics = calculate_metrics(actual, predicted)
    naive_metrics = calculate_metrics(actual, naive)

    summary_rows.append({
        "forecast_horizon": "Overall",
        "observations": len(predictions),
        "MAE": model_metrics["MAE"],
        "RMSE": model_metrics["RMSE"],
        "Seasonal_Naive_MAE": naive_metrics["MAE"],
        "Seasonal_Naive_RMSE": naive_metrics["RMSE"],
        "MAE_Improvement_Pct": (
            (1 - model_metrics["MAE"] / naive_metrics["MAE"]) * 100
            if naive_metrics["MAE"] > 0 else np.nan
        ),
        "RMSE_Improvement_Pct": (
            (1 - model_metrics["RMSE"] / naive_metrics["RMSE"]) * 100
            if naive_metrics["RMSE"] > 0 else np.nan
        ),
    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUTS / "rolling_backtest_metrics.csv", index=False)

    print("\nROLLING-ORIGIN BACKTEST RESULTS")
    print(summary.round(3).to_string(index=False))

    print("\nFiles saved:")
    print(OUTPUTS / "rolling_backtest_predictions.csv")
    print(OUTPUTS / "rolling_backtest_metrics.csv")


if __name__ == "__main__":
    run_rolling_backtest()
