import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from utils import TimeSeriesDataset, safe_time_split
from model import ForecastingModel

def train_and_evaluate(grouped, min_needed, model_dir, device, window, asplit, batch_size, epochs, lr, model_type, input_dim, hidden_dim, num_layers, output_dim, dropout, target_col):
    all_true = []
    all_pred = []

    for (school, meal), g in grouped:
        series_values = g[target_col].values.astype("float32").reshape(-1, 1)
        n_points = len(series_values)
        print(f"Training model for school={school!r}, meal_type={meal!r} | points={n_points}")

        if n_points < min_needed:
            print(f"[skip] Too few points ({n_points}) for WINDOW={window}. Need at least {min_needed}.")
            continue

        try:
            split_idx = safe_time_split(series_values, asplit, window)
        except ValueError as e:
            print(f"[skip] {school!r}/{meal!r} due to split error: {e}")
            continue

        train_raw = series_values[:split_idx]
        test_raw = series_values[split_idx:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        train_ds = TimeSeriesDataset(train_scaled, window)
        test_ds = TimeSeriesDataset(test_scaled, window)

        if len(train_ds) == 0:
            print(f"[skip] No train windows for WINDOW={window} in {school!r}/{meal!r}.")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = ForecastingModel(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(1, epochs + 1), desc=f"Training {model_type} for {school} / {meal}", leave=False):
            model.train()
            for xb, yb in train_loader:
                xb = xb.unsqueeze(-1).to(device)
                yb = yb.unsqueeze(-1).to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        eval_loader = test_loader if len(test_ds) > 0 else train_loader
        model.eval()
        preds_scaled, ys_scaled = [], []
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.unsqueeze(-1).to(device)
                yb = yb.unsqueeze(-1).to(device)
                p = model(xb)
                preds_scaled.append(p.detach().cpu().numpy())
                ys_scaled.append(yb.detach().cpu().numpy())

        if len(preds_scaled) > 0:
            y_pred = scaler.inverse_transform(np.vstack(preds_scaled))
            y_true = scaler.inverse_transform(np.vstack(ys_scaled))
            all_true.append(y_true.reshape(-1, 1))
            all_pred.append(y_pred.reshape(-1, 1))

        school_safe = str(school).replace(" ", "_").replace("/", "_")
        meal_safe = str(meal).replace(" ", "_").replace("/", "_")
        model_path = os.path.join(model_dir, f"{model_type}_{school_safe}_{meal_safe}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[saved] {model_path}")

    return all_true, all_pred
