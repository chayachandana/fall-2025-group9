#%%
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import pickle
import importlib, utils
importlib.reload(utils)
from utils import (
    seed_everything, load_and_aggregate_district, safe_time_split, TimeSeriesDataset
)
from model import FeedForwardRegressor
import faulthandler
faulthandler.enable()

current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(current_dir, '..', 'Data', 'Output', 'meals_combined.csv')
DATE_COL   = "date"
TARGET_COL = "production_cost_total"

WINDOW     = 7
ASPLIT     = 0.7
SEED       = 42
results_dir = os.path.join(current_dir, '..', '..', 'results')
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, '..', '..', '..', 'demo', 'images', 'univariate_plots')

EPOCHS_FNN = 100
LR_FNN     = 0.001
HIDDEN     = 256
DROPOUT    = 0.25

def ds_to_numpy(ds, window: int):
    N = len(ds)
    X = np.empty((N, window), dtype=np.float32)
    y = np.empty((N,), dtype=np.float32)
    for i in range(N):
        Xi, yi = ds[i]
        if isinstance(Xi, torch.Tensor):
            Xi = Xi.numpy()
        if isinstance(yi, torch.Tensor):
            yi = float(yi.item())
        X[i] = Xi
        y[i] = yi
    return X, y

def invert_with(scaler, vec_1d):
    return scaler.inverse_transform(np.asarray(vec_1d).reshape(-1,1)).reshape(-1)

def eval_report(name, y_true_np, y_pred_np):
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_np, y_pred_np)
    print(f"{name:12s} | MSE={mse:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}")
    return (name, mse, rmse, r2)

seed_everything(SEED)
print("Using utils from:", getattr(utils, "__file__", "<unknown>"))
print("Using file:", os.path.abspath(CSV_PATH))

dates, values, DATE, TARGET = load_and_aggregate_district(CSV_PATH, DATE_COL, TARGET_COL)
print(f"DATE_COL={DATE} | TARGET_COL={TARGET} | total_days={len(values)}")

min_needed = 2 * (WINDOW + 1) + 1
if len(values) < min_needed:
    new_window = max(1, (len(values) - 3) // 2)
    print(f"[info] Not enough points for WINDOW={WINDOW}. Setting WINDOW -> {new_window}")
    WINDOW = new_window

split = safe_time_split(values, ASPLIT, WINDOW)
train_raw, test_raw = values[:split], values[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)

train_ds = TimeSeriesDataset(train_scaled, WINDOW)
test_ds  = TimeSeriesDataset(test_scaled,  WINDOW)
print(f"train_windows={len(train_ds)} | test_windows={len(test_ds)}")

Xtr, ytr = ds_to_numpy(train_ds, WINDOW)
Xe,  ye  = ds_to_numpy(test_ds,  WINDOW)
y_true = invert_with(scaler, ye)

results = []

lin = LinearRegression()
lin.fit(Xtr, ytr)
pred_lin = invert_with(scaler, lin.predict(Xe))
results.append(eval_report("LinearReg", y_true, pred_lin))
with open(os.path.join(results_dir, "linear_regression.pkl"), "wb") as f:
    pickle.dump(lin, f)
print(" Saved Linear Regression model and plot")

plt.figure(figsize=(6,6))
plt.scatter(y_true, pred_lin, alpha=0.6)
lo = float(min(np.min(y_true), np.min(pred_lin)))
hi = float(max(np.max(y_true), np.max(pred_lin)))
plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title("Linear Regression Predicted vs Actual")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(plots_dir, "linear_regression.png"), bbox_inches='tight', dpi=300)
plt.close()

xgb = XGBRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=1, 
)
xgb.fit(Xtr, ytr)
pred_xgb_scaled = xgb.predict(Xe)
pred_xgb = invert_with(scaler, pred_xgb_scaled)
results.append(eval_report("XGBoost", y_true, pred_xgb))

plt.figure(figsize=(6,6))
plt.scatter(y_true, pred_xgb, alpha=0.6)
lo = float(min(np.min(y_true), np.min(pred_xgb)))
hi = float(max(np.max(y_true), np.max(pred_xgb)))
plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title("XGBoost Predicted vs Actual")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(plots_dir, "xgboost_model.png"), bbox_inches='tight', dpi=300)
xgb.save_model(os.path.join(results_dir, "xgboost_model.json"))
print("Saved XGBoost model and plot")
plt.close()

#%%
print("=== Entering FNN Block ===")
device = torch.device("cpu")
fnn = FeedForwardRegressor(in_dim=WINDOW, hidden=HIDDEN, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(fnn.parameters(), lr=LR_FNN)
crit = nn.MSELoss()

Xtr_t = torch.from_numpy(Xtr).to(device)
ytr_t = torch.from_numpy(ytr).unsqueeze(-1).to(device)
Xe_t  = torch.from_numpy(Xe).to(device)

fnn.train()
for epoch in range(EPOCHS_FNN):
    opt.zero_grad()
    out = fnn(Xtr_t)
    loss = crit(out, ytr_t)
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"FNN Epoch {epoch}/{EPOCHS_FNN} - Loss: {loss.item():.6f}")

fnn.eval()
with torch.no_grad():
    pred_fnn_scaled = fnn(Xe_t).cpu().numpy().reshape(-1)
pred_fnn = invert_with(scaler, pred_fnn_scaled)
results.append(eval_report("FNN", y_true, pred_fnn))

plt.figure(figsize=(6,6))
plt.scatter(y_true, pred_fnn, alpha=0.6)
lo = float(min(np.min(y_true), np.min(pred_fnn)))
hi = float(max(np.max(y_true), np.max(pred_fnn)))
plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title("FNN Predicted vs Actual")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(plots_dir, "fnn_model.png"), bbox_inches='tight', dpi=300)
torch.save(fnn.state_dict(), os.path.join(results_dir, "fnn_model.pth"))
print("Saved FNN model and plot")
plt.close()

print("\n=== Model Comparison (lower RMSE/MSE better; higher R² better) ===")
for name, mse, rmse, r2 in results:
    print(f"{name:12s}  MSE={mse:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

print("\n All models trained and saved successfully.")
