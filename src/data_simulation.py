import numpy as np
import pandas as pd
from src.config import DATA_RAW, RANDOM_STATE

def generate_retail_data(days=540):
    #\"\"\"Create reproducible daily sales for several products and stores.\"\"\"
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
                          periods=days, freq="D")
    products = [
        ("P001", "Beverages", 35, 0.12),
        ("P002", "Snacks", 22, 0.18),
        ("P003", "Personal Care", 48, 0.10),
        ("P004", "Household", 65, 0.08),
        ("P005", "Dairy", 30, 0.15),
        ("P006", "Staples", 55, 0.07),
    ]
    stores = [("S001", "North"), ("S002", "South"), ("S003", "West")]
    rows = []
    for store_id, region in stores:
        store_factor = {"North": 1.00, "South": 0.90, "West": 1.12}[region]
        for product_id, category, price, noise in products:
            base = rng.uniform(18, 65) * store_factor
            for i, date in enumerate(dates):
                weekly = 1 + 0.16 * (date.dayofweek in [5, 6])
                annual = 1 + 0.18 * np.sin(2 * np.pi * i / 365.25)
                trend = 1 + 0.00035 * i
                promotion = int(rng.random() < 0.07)
                promo_factor = 1.35 if promotion else 1.0
                demand = base * weekly * annual * trend * promo_factor
                units = max(0, int(round(rng.normal(demand, max(2, demand * noise)))))
                rows.append({
                    "date": date, "store_id": store_id, "region": region,
                    "product_id": product_id, "category": category,
                    "unit_price": price, "promotion": promotion,
                    "units_sold": units,
                    "revenue": round(units * price, 2)
                })
    df = pd.DataFrame(rows)
    path = DATA_RAW / "retail_sales.csv"
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    data = generate_retail_data()
    print(f"Created {len(data):,} rows at data/raw/retail_sales.csv")
"""

files["src/preprocessing.py"] = """
import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED

REQUIRED = ["date", "store_id", "product_id", "category", "unit_price",
            "promotion", "units_sold", "revenue"]

def load_and_clean(path=None):
    path = path or DATA_RAW / "retail_sales.csv"
    df = pd.read_csv(path)
    missing_columns = set(REQUIRED) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "store_id", "product_id", "units_sold"])
    df = df.drop_duplicates()
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").clip(lower=0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").clip(lower=0)
    df = df.dropna(subset=["units_sold", "unit_price"])
    df["revenue"] = df["units_sold"] * df["unit_price"]
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "clean_sales.csv", index=False)
    return df

if __name__ == "__main__":
    clean = load_and_clean()
    print(clean.info())
"""

files["src/features.py"] = """
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
