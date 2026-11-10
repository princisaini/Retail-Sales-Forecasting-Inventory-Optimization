import pandas as pd

GROUPS = ["store_id", "product_id"]

def add_features(df):
   # \"\"\"Create leakage-safe lag/rolling features using past observations only.\"\"\"
    df = df.sort_values(GROUPS + ["date"]).copy()
    grouped = df.groupby(GROUPS, sort=False)["units_sold"]
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    df["rolling_mean_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    )
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["promotion"] = df["promotion"].fillna(0).astype(int)
    return df

FEATURES = [
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "day_of_year", "is_weekend",
    "promotion", "unit_price"
]