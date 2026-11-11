# Retail Sales Forecasting & Inventory Optimization System

An end-to-end retail analytics project that combines demand forecasting, inventory replenishment recommendations, model evaluation, and an interactive Streamlit dashboard.

## Project Overview

Retail businesses need to anticipate product demand while maintaining appropriate inventory levels. Understocking can lead to stockouts and lost sales, while overstocking increases holding costs and ties up working capital.

This project demonstrates a data-driven approach to:
- Analyze historical retail sales.
- Forecast product demand for the next seven days.
- Evaluate forecasting performance against a seasonal-naive baseline.
- Identify potential stockout and excess-inventory risks.
- Recommend replenishment quantities.
- Monitor results through an interactive dashboard.

**Note:** The dataset and inventory scenarios in this project are simulated for demonstration and analysis. They do not represent live retail transactions or actual store inventory.

## Key Features

- Synthetic retail sales data generation.
- Data cleaning and preprocessing.
- Time-based feature engineering, including lag and rolling features.
- Random Forest demand forecasting.
- Seven-day recursive demand forecasting.
- Seasonal-naive benchmark comparison.
- Rolling-origin recursive backtesting.
- Inventory optimization using lead-time demand, safety stock, reorder points, and order-up-to levels.
- Stockout-risk and excess-inventory indicators.
- Interactive dashboard with store, product, and category filters.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model persistence | Joblib |

## Project Structure

```text
Retail-Sales-Forecasting-Inventory-Optimization/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│   ├── actual_vs_predicted.png
│   ├── category_revenue.png
│   ├── sales_trend.png
│   └── seven_day_forecast.png
│
├── models/
│   └── demand_model.joblib
│
├── outputs/
│   ├── holdout_predictions.csv
│   ├── inventory_recommendations.csv
│   ├── inventory_recommendations_sorted.csv
│   ├── model_metrics.txt
│   ├── next_7_day_forecast.csv
│   ├── rolling_backtest_metrics.csv
│   └── rolling_backtest_predictions.csv
│
├── src/
│   ├── backtesting.py
│   ├── config.py
│   ├── data_simulation.py
│   ├── features.py
│   ├── forecasting.py
│   ├── inventory.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Forecasting Methodology

A Random Forest Regressor is trained using historical sales and time-based demand features.

Feature engineering includes:
- Lagged sales values.
- Rolling demand statistics.
- Calendar features.
- Day-of-week and weekend indicators.

The project uses chronological data splitting to preserve time order and reduce temporal leakage.

### Model Evaluation

The one-step-ahead chronological holdout evaluation produced:

| Metric | Random Forest | Seasonal Naive |
|---|---:|---:|
| MAE | 6.02 | 9.11 |
| RMSE | 8.34 | 13.07 |

- MAE improvement: **33.97%**
- RMSE improvement: **36.17%**

The seasonal-naive model uses demand from seven days earlier as its prediction.

### Recursive Seven-Day Backtesting

A separate rolling-origin backtest evaluates multi-step forecasting. At each forecast step, predicted demand is fed into subsequent lag features rather than using actual future sales.

Results across 12 historical forecast origins:

| Metric | Random Forest | Seasonal Naive |
|---|---:|---:|
| Overall MAE | 6.540 | 8.949 |
| Overall RMSE | 9.694 | 12.793 |

- MAE improvement: **26.92%**
- RMSE improvement: **24.22%**
- Forecast horizons: 1–7 days.
- Evaluation observations: 1,512.

The model outperformed the seasonal-naive benchmark across all seven forecast horizons in this backtest.

## Inventory Optimization

The inventory module calculates replenishment indicators using forecast demand and inventory planning parameters.

Key concepts:

- **Lead-time demand:** Expected demand during supplier lead time.
- **Safety stock:** Additional stock maintained to protect against demand uncertainty.
- **Reorder point:** Lead-time demand plus safety stock.
- **Order-up-to level:** Target inventory level used to calculate replenishment quantity.

The recommendation logic distinguishes between:
- Reorder required.
- Stockout risk.
- Healthy inventory.
- Excess inventory.

The recommended order quantity is zero when replenishment is not required.

**Inventory data limitation:** On-hand stock is simulated using deterministic scenarios (low, healthy, and excess stock). Recommendations demonstrate the decision logic and are not purchase orders.

## Interactive Dashboard

Run the dashboard to explore:
- Historical revenue and sales trends.
- Seven-day demand forecasts.
- Forecast-versus-actual model evaluation.
- Revenue by category.
- Reorder alerts and stockout-risk indicators.
- Inventory recommendations.

Store, product, and category filters update the dashboard views and metrics.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/princisaini/Retail-Sales-Forecasting-Inventory-Optimization.git
cd Retail-Sales-Forecasting-Inventory-Optimization
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the complete pipeline

```bash
python main.py
```

This generates the processed data, model outputs, forecasts, inventory recommendations, and visualizations.

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal.

## Limitations

- Sales data is synthetically generated.
- Inventory levels and scenarios are simulated.
- The model is not connected to live point-of-sale or inventory systems.
- Supplier lead times and inventory planning parameters are demonstration assumptions.
- Backtesting results reflect the generated dataset and should not be interpreted as guaranteed real-world performance.

## Future Improvements

- Integrate real retail sales and inventory datasets.
- Evaluate additional forecasting algorithms.
- Add automated hyperparameter optimization.
- Incorporate dynamic supplier lead times and demand uncertainty.
- Add inventory holding, ordering, and stockout cost optimization.
- Deploy the dashboard with scheduled data updates.

## Author

**Princi Saini**

B.Tech — Electronics and Computer Science Engineering

GitHub: [princisaini](https://github.com/princisaini)

---

*Built as an end-to-end retail analytics and decision-support project.*
### Model Evaluation

The Random Forest Regressor achieved an MAE of **6.02 units** and an RMSE of **8.34 units** on a chronological holdout dataset containing 1,908 observations from 27 July to 9 November 2026.

Compared with a seasonal-naive baseline based on sales from the previous week, the model reduced MAE by **33.97%** and RMSE by **36.17%**.

The evaluation uses one-step-ahead predictions with observed historical lag features. Recursive multi-step forecasting is implemented separately for the seven-day demand forecast and requires dedicated backtesting for a direct assessment of its performance.
