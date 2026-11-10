# Phase 3 changes

- Added a seasonal-naive baseline using lag-7 observed sales on the chronological holdout.
- Saved seasonal_naive_units alongside actual and Random Forest holdout predictions.
- Added filtered holdout actual-vs-predicted dashboard chart and MAE/RMSE comparisons.
- Fixed replenishment quantity: recommended_order_units is zero unless reorder_required is true.
- Added days_of_inventory, stockout_risk, excess_inventory_units and excess_inventory.
- Added dashboard inventory risk filters and KPIs.

## Important evaluation limitation
The holdout evaluation is one-step-ahead: lag features in the test set use observed prior sales.
It is not equivalent to a recursive seven-day backtest. Run a rolling-origin / recursive
backtest before making multi-step forecast accuracy claims.

## Run
1. Activate your virtual environment.
2. Run `python main.py` to regenerate all outputs.
3. Run `streamlit run app/streamlit_app.py`.
