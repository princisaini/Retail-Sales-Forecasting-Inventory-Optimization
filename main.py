from src.data_simulation import generate_retail_data
from src.preprocessing import load_and_clean
from src.forecasting import train_and_evaluate, forecast_next_days
from src.inventory import create_inventory_recommendations
from src.visualization import create_charts
from src.config import OUTPUTS

def main():
    print("1/6 Generating virtual retail sales data...")
    generate_retail_data()
    print("2/6 Cleaning data...")
    df = load_and_clean()
    print(f"   Clean rows: {len(df):,}")
    print("3/6 Training and evaluating demand model...")
    model, metrics, holdout = train_and_evaluate(df)
    print(f"   MAE: {metrics['MAE']:.2f} units | RMSE: {metrics['RMSE']:.2f} units")
    print("4/6 Forecasting next 7 days...")
    forecast = forecast_next_days(df, model, horizon=7)
    print("5/6 Creating inventory recommendations...")
    inventory = create_inventory_recommendations(df, forecast)
    print("6/6 Saving business charts...")
    create_charts(df, holdout, forecast, inventory)
    print("\\nCompleted. Outputs saved in outputs/ and images/.")
    print(inventory.head(10).to_string(index=False))
    print(f"Metrics saved in {OUTPUTS / 'model_metrics.txt'}")
    with open(OUTPUTS / "model_metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\\n")

if __name__ == "__main__":
    main()