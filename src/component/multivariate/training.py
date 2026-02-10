import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pickle
from plot import plot_actual_vs_predicted
from model import FeedForwardNN, ForecastingModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, '..', '..', 'results')



def train_linear_regression(X_train, y_train, X_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    print(f"Linear Regression → MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}, MAE: {mae_lr:.4f}")
    lr_filepath = os.path.join(results_dir, "linear_regression.pkl")
    with open(lr_filepath, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"Saved LR model to {lr_filepath}")
    plot_actual_vs_predicted(y_test, y_pred_lr, "Linear Regression", "multivariate/plots/linear_regression.png")

def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print(f"XGBoost → MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}, MAE: {mae_xgb:.4f}")
    xgb_filepath = os.path.join(results_dir, "xgboost_model.json")
    xgb_model.save_model(xgb_filepath)
    plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost", "multivariate/plots/xgboost_model.png")

def train_fnn(X_train, y_train, X_test, y_test, features):
    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    y_tensor_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_tensor_test = torch.tensor(X_test, dtype=torch.float32)

    fnn_model = FeedForwardNN(input_dim=len(features), hidden_dim=256, output_dim=1)
    criterion_fnn = nn.MSELoss()
    optimizer_fnn = torch.optim.Adam(fnn_model.parameters(), lr=0.0005)

    epochs_fnn = 60
    for epoch in range(epochs_fnn):
        fnn_model.train()
        optimizer_fnn.zero_grad()
        preds_fnn = fnn_model(X_tensor_train)
        loss_fnn = criterion_fnn(preds_fnn, y_tensor_train)
        loss_fnn.backward()
        optimizer_fnn.step()
        if epoch % 5 == 0:
            print(f"FNN Epoch {epoch}/{epochs_fnn} — Loss: {loss_fnn.item():.6f}")

    fnn_model.eval()
    with torch.no_grad():
        y_pred_fnn = fnn_model(X_tensor_test).numpy().flatten()
    mse_fnn = mean_squared_error(y_test, y_pred_fnn)
    r2_fnn = r2_score(y_test, y_pred_fnn)
    mae_fnn = mean_absolute_error(y_test, y_pred_fnn)
    print(f"FNN → MSE: {mse_fnn:.4f}, R²: {r2_fnn:.4f}, MAE: {mae_fnn:.4f}")
    fnn_filepath = os.path.join(results_dir, "fnn_model.pth")
    torch.save(fnn_model.state_dict(), fnn_filepath)
    plot_actual_vs_predicted(y_test, y_pred_fnn, "FNN", "multivariate/plots/fnn_model.png")

def train_gru(preprocessed_data, features, target):
    dataset = SchoolMealDataset(preprocessed_data, features, target)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    MODEL_TYPE = "GRU"
    INPUT_DIM = len(features)
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    OUTPUT_DIM = 1
    DROPOUT = 0.25
    LEARNING_RATE = 0.001
    EPOCHS = 60

    model = ForecastingModel(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS), desc=f"Training {MODEL_TYPE}"):
        model.train()
        running_loss = 0
        for X_batch, y_batch, lengths in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch, lengths)
            y_true = torch.stack([y_batch[i, l - 1] for i, l in enumerate(lengths)]).unsqueeze(1)
            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss / len(train_loader):.6f}")

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            preds = model(X_batch, lengths)
            y_true = torch.stack([y_batch[i, l - 1] for i, l in enumerate(lengths)]).unsqueeze(1)
            y_true_all.extend(y_true.numpy())
            y_pred_all.extend(preds.numpy())

    mse = mean_squared_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)

    print(f"\n Final Results for {MODEL_TYPE}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    torch.save(model.state_dict(), os.path.join(results_dir, f"{MODEL_TYPE}.pth"))
    plot_actual_vs_predicted(np.array(y_true_all), np.array(y_pred_all), MODEL_TYPE, f"multivariate/plots/{MODEL_TYPE}.png")

class SchoolMealDataset(Dataset):
    def __init__(self, df, features, target):
        self.data = []
        groups = df.groupby(["school_name", "meal_type"])
        for _, group in groups:
            group = group.sort_values("date")
            X = torch.tensor(group[features].values, dtype=torch.float32)
            y = torch.tensor(group[target].values, dtype=torch.float32)
            self.data.append((X, y))
        self.max_len = max(len(x[0]) for x in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return X, y, len(X)

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    Xs, ys, lengths = zip(*batch)
    X_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    return X_padded, y_padded, torch.tensor(lengths)
