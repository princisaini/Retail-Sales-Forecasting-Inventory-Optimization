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