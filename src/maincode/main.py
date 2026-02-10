import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'component', 'univariate')))
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from utils import seed_everything, load_and_aggregate_district, safe_time_split
from forecasting import forecast_future_dates
from plot import plot_actual_vs_predicted, plot_train_test_forecast
from training import train_and_evaluate

current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(current_dir, '..', 'Data', 'Output', 'meals_combined.csv')
DATE_COL   = "date"

TARGET_COL = "production_cost_total"

# Model and training hyperparameters
WINDOW     = 3; ASPLIT     = 0.6; MODEL_TYPE = "LSTM"; HIDDEN_DIM = 256
INPUT_DIM  = 1; OUTPUT_DIM = 1; NUM_LAYERS = 4; DROPOUT= 0.25
EPOCHS     = 100; BATCH_SIZE = 64; LR= 0.001; SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED)
torch.manual_seed(SEED)

def main():
    model_dir = os.path.join(current_dir, '..', 'component', 'univariate', f"{MODEL_TYPE}_models")
    results_dir = os.path.join(current_dir, '..', 'results')
    plots_dir = os.path.join(current_dir, '..', '..', 'demo', 'images', 'univariate_plots')
    for d in (model_dir,results_dir,plots_dir): os.makedirs(d,exist_ok=True)

    dates, values, _, _ = load_and_aggregate_district(
        CSV_PATH=CSV_PATH, DATE_COL=DATE_COL, TARGET_COL=TARGET_COL, dayfirst="auto", debug=True
    )
    records = [(school, meal, pd.to_datetime(dt), float(v)) for (school, meal, dt), v in zip(dates, values.reshape(-1))]
    df_series = pd.DataFrame(records, columns=["school_name", "meal_type", DATE_COL, TARGET_COL])
    df_series = df_series.sort_values(["school_name", "meal_type", DATE_COL]).reset_index(drop=True)
    grouped = df_series.groupby(["school_name", "meal_type"], sort=True)

    min_needed = 2 * (WINDOW + 1) + 1
    all_true, all_pred = train_and_evaluate(
        grouped, min_needed, model_dir, device, WINDOW, ASPLIT, BATCH_SIZE, EPOCHS, LR,
        MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT, TARGET_COL
    )

    if len(all_true) > 0:
        all_true_arr = np.vstack(all_true)
        all_pred_arr = np.vstack(all_pred)
        overall_mse  = mean_squared_error(all_true_arr.ravel(), all_pred_arr.ravel())
        overall_rmse = float(np.sqrt(overall_mse))
        overall_r2   = r2_score(all_true_arr.ravel(), all_pred_arr.ravel())
        print(f"\n=== OVERALL RESULTS for {MODEL_TYPE} (all schools & meals) ===")
        print(f"MSE : {overall_mse:.4f}")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"R^2 : {overall_r2:.4f}")
        plot_actual_vs_predicted(all_true_arr, all_pred_arr, MODEL_TYPE, os.path.join(plots_dir, f"{MODEL_TYPE}.png"))

    all_forecasts = []
    forecast_true_all = []
    forecast_pred_all = []
    unique_combinations = df_series[['school_name', 'meal_type']].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        school, meal = row['school_name'], row['meal_type']
        try:
            df_forecast, bt_true, bt_pred = forecast_future_dates(csv_path=CSV_PATH,date_col=DATE_COL,target_col=TARGET_COL,window=WINDOW,
                                    asplit=ASPLIT,k_steps=10,model_type=MODEL_TYPE,model_path=os.path.join(model_dir, f"{MODEL_TYPE}.pth"),
                                    hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,dropout=DROPOUT,school_name=school,meal_type=meal)
            all_forecasts.append(df_forecast)
            if bt_true is not None:
                forecast_true_all.append(bt_true)
                forecast_pred_all.append(bt_pred)

        except Exception as e:
            print(f"[error] Forecasting failed for school={school!r}, meal_type={meal!r}: {e}")

    if len(all_forecasts) > 0:
        df_all_forecast = pd.concat(all_forecasts, ignore_index=True)
        output_path = os.path.join(results_dir, "all_school_meal_forecasts.csv")
        df_all_forecast.to_csv(output_path, index=False)
        print(f"\n[saved] Combined 10-day forecasts for all schools/meals -> {output_path}")
        print(df_all_forecast.head(20))
        if len(forecast_true_all) > 0:
            y_true_bt = np.concatenate(forecast_true_all)
            y_pred_bt = np.concatenate(forecast_pred_all)
            bt_mse  = mean_squared_error(y_true_bt, y_pred_bt)
            bt_rmse = float(np.sqrt(bt_mse))
            bt_r2   = r2_score(y_true_bt, y_pred_bt)
            print("\n=== OVERALL 10-STEP FORECAST BACKTEST (all schools & meals) ===")
            print(f"Forecast MSE : {bt_mse:.4f}\nForecast RMSE: {bt_rmse:.4f}\nForecast R^2: {bt_r2:.4f}")
        else:
            print("\n[warn] No backtest forecast metrics collected (series too short).")
        
        # ---- Example train/test/forecast plot for first school+meal ----
        ex=unique_combinations.iloc[1]
        ex_school,ex_meal=ex["school_name"],ex["meal_type"]
        df_ex=df_series[(df_series["school_name"]==ex_school)&(df_series["meal_type"]==ex_meal)].copy()
        if df_ex.empty:
            print(f"[warn] No data for example series {ex_school!r}/{ex_meal!r}")
        else:
            vals = df_ex[TARGET_COL].values.astype("float32").reshape(-1, 1)
            split_idx = safe_time_split(vals, ASPLIT, WINDOW)

            df_ex_forecast = df_all_forecast[(df_all_forecast["school_name"] == ex_school) &(df_all_forecast["meal_type"] == ex_meal)].sort_values("forecast_date")

            plot_train_test_forecast(dates=df_ex[DATE_COL].values,values=df_ex[TARGET_COL].values,
                        split_idx=split_idx,forecast_dates=df_ex_forecast["forecast_date"].values,
                        forecast_values=df_ex_forecast[TARGET_COL].values,title=f"{MODEL_TYPE} Train/Test/Forecast - {ex_school} ({ex_meal})",
                        plot_path=os.path.join(plots_dir, f"{MODEL_TYPE}_train_test_forecast_example.png"))
if __name__ == "__main__":
    main()
