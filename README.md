### Model Evaluation

The Random Forest Regressor achieved an MAE of **6.02 units** and an RMSE of **8.34 units** on a chronological holdout dataset containing 1,908 observations from 27 July to 9 November 2026.

Compared with a seasonal-naive baseline based on sales from the previous week, the model reduced MAE by **33.97%** and RMSE by **36.17%**.

The evaluation uses one-step-ahead predictions with observed historical lag features. Recursive multi-step forecasting is implemented separately for the seven-day demand forecast and requires dedicated backtesting for a direct assessment of its performance.
