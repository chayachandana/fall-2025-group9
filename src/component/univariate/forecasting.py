import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import seed_everything, TimeSeriesDataset, load_and_aggregate_district, safe_time_split
from model import ForecastingModel


# Hyperparameters (defaults; can be overridden via function args)
CSV_PATH = "src/Data/Output/meals_combined.csv"
DATE_COL = "date"
TARGET_COL = "production_cost_total"

MODEL_TYPE = "LSTM"
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.25
WINDOW = 3
ASPLIT = 0.6
K_STEPS = 10
MODEL_PATH = f"univariate/LSTM_models/{MODEL_TYPE}.pth"

def next_days(last_date: pd.Timestamp, k: int) -> pd.DatetimeIndex:
    """Return the next k business days after last_date."""
    start = pd.to_datetime(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start, periods=k)

def last_date_from_dates_array(dates_array) -> pd.Timestamp:
    """Get the last date from a numpy array of dates.
    Handles both simple datetime arrays and tuples like
    (school_name, meal_type, Timestamp).
    """
    last_item = dates_array[-1]
    if isinstance(last_item, tuple) and len(last_item) >= 3:
        return pd.to_datetime(last_item[2])
    return pd.to_datetime(last_item)

def forecast_future_dates(
    csv_path: str = CSV_PATH,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
    window: int = WINDOW,
    asplit: float = ASPLIT,
    k_steps: int = K_STEPS,
    model_type: str = MODEL_TYPE,
    model_path: str = MODEL_PATH,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    school_name=None,
    meal_type=None,
) -> pd.DataFrame:
    """Forecast the next k_steps of production cost for a given school and meal type.

    This function:
    1. Loads the aggregated (school, meal, date) series from CSV.
    2. Filters to the requested (school_name, meal_type).
    3. Recomputes the train split and scaler just like in main.py.
    4. Loads the trained LSTM for that school+meal.
    5. Generates k_steps future daily forecasts (business days).
    """
    if school_name is None or meal_type is None:
        raise ValueError("You must pass both school_name and meal_type to forecast_future_dates.") 
     # Load and aggregate data at (school, meal, date) level
    dates, values, _, _ = load_and_aggregate_district(
        CSV_PATH=csv_path,
        DATE_COL=date_col,
        TARGET_COL=target_col,
        dayfirst="auto",
        debug=False,
    )

    if len(values) < window + 1:
        raise ValueError(f"Overall series too short (n={len(values)}) for WINDOW={window}.")  # unlikely
    # Rebuild tidy DataFrame
    records = []
    for (school, meal, dt), v in zip(dates, values.reshape(-1)):
        records.append((school, meal, pd.to_datetime(dt), float(v)))

    df_series = pd.DataFrame(records, columns=["school_name", "meal_type", date_col, target_col])
    df_series = df_series.sort_values(["school_name", "meal_type", date_col]).reset_index(drop=True)

    # Filter for the requested school and meal
    df_sm = df_series[(df_series["school_name"] == school_name) & (df_series["meal_type"] == meal_type)].copy()
    if df_sm.empty:
        raise ValueError(f"No data found for school={school_name!r}, meal_type={meal_type!r}")
    values_sm = df_sm[target_col].values.astype("float32").reshape(-1, 1)
    dates_sm  = df_sm[date_col].values
    if len(values_sm) < window + 1:
        raise ValueError(
            f"Series for {school_name!r}/{meal_type!r} too short (n={len(values_sm)}) for WINDOW={window}."
        )
    split_idx = safe_time_split(values_sm, asplit, window)
    train_raw = values_sm[:split_idx]
    scaler = MinMaxScaler()
    scaler.fit(train_raw)
    full_scaled = scaler.transform(values_sm).ravel()

    # Load the trained model weights for this school+meal
    school_safe = str(school_name).replace(" ", "_").replace("/", "_")
    meal_safe   = str(meal_type).replace(" ", "_").replace("/", "_")
    base_dir = os.path.dirname(model_path) or "."
    model_file = f"{model_type}_{school_safe}_{meal_safe}.pth"
    model_path_effective = os.path.join(base_dir, model_file)

    if not os.path.exists(model_path_effective):
        raise FileNotFoundError(f"Saved model not found for {school_name!r}/{meal_type!r}: {model_path_effective}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastingModel(
        model_type=model_type,
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
    ).to(device)

    state = torch.load(model_path_effective, map_location=device)
    model.load_state_dict(state)
    model.eval()
    bt_true = None
    bt_pred = None

    n_points = len(values_sm)
    if n_points >= window + k_steps + 1:
        # index where pseudo-future starts (last k_steps)
        hist_len = n_points - k_steps
        last_window_bt = full_scaled[hist_len - window : hist_len].astype(np.float32).copy()
        preds_scaled_bt = []
        with torch.no_grad():
            for _ in range(k_steps):
                xb_bt = torch.from_numpy(last_window_bt).view(1, window, 1).to(device)
                yhat_bt = model(xb_bt).detach().cpu().numpy().reshape(-1)[0]
                preds_scaled_bt.append(yhat_bt)
                # slide the window: drop oldest, append new prediction
                last_window_bt = np.roll(last_window_bt, -1)
                last_window_bt[-1] = yhat_bt
        # backtest predictions in original scale
        preds_bt = scaler.inverse_transform(
            np.array(preds_scaled_bt).reshape(-1, 1)
        ).ravel()
        # true last k_steps values
        true_bt = values_sm[hist_len:].reshape(-1)
        bt_true = true_bt
        bt_pred = preds_bt
        mse_bt  = mean_squared_error(true_bt, preds_bt)
        rmse_bt = float(np.sqrt(mse_bt))
        r2_bt   = r2_score(true_bt, preds_bt)
        print(
            f"Forecasting for {school_name!r}/{meal_type!r}")
    else:
        print(f"Forecasting Skipped for {school_name!r}/{meal_type!r}: "
            f"series too short for window={window}, k_steps={k_steps}")     
    last_window = full_scaled[-window:].astype(np.float32).copy()
    preds_scaled = []
    with torch.no_grad():
        for _ in range(k_steps):
            xb = torch.from_numpy(last_window).view(1, window, 1).to(device)
            yhat = model(xb).detach().cpu().numpy().reshape(-1)[0]
            preds_scaled.append(yhat)
            last_window = np.roll(last_window, -1)
            last_window[-1] = yhat
    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).ravel()

    # Compute future dates starting from last actual date in this series
    last_date = df_sm[date_col].max()
    fdates = next_days(last_date, k_steps)
    out = pd.DataFrame(
        {
            "school_name": school_name,
            "meal_type": meal_type,
            "forecast_date": fdates,
            "step_ahead": np.arange(1, k_steps + 1),
            target_col: preds,
        }
    )
    print(f"forecasting: {len(out)} rows for {school_name!r} / {meal_type!r}")
    return out, bt_true, bt_pred

if __name__ == "__main__":
    df, bt_true, bt_pred = forecast_future_dates()
    print(df.head())