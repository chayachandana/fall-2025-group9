
# def preprocessing(df):
#     return preprocessed_data
# utils.py
import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed=42):
    """Ensures full reproducibility for PyTorch, NumPy, and Python."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

class EarlyStopping:
    """Stop training when validation loss doesn't improve after 'patience' epochs."""
    def __init__(self, model_name, patience=5, delta=0, verbose=False):
        self.model_name = model_name
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss  # lower loss is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss improved. Saving model '{self.model_name}.pt' ...")

        os.makedirs("save", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("save", f"{self.model_name}.pt"))
        self.val_score = val_loss


def preprocessing(df, seq_len=10, threshold=0.99):
    """
    Preprocesses the meals dataset for time series forecasting using LSTM/GRU.
    Keeps categorical and numeric features.
    """

    df = df.fillna(method='ffill').fillna(method='bfill')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    # --- Clean up 'production_cost_total' if it exists ---
    if 'production_cost_total' in df.columns:
        df['production_cost_total'] = (
            df['production_cost_total']
            .replace({'\$': '', ',': ''}, regex=True)
            .astype(float)
        )

    # Encode meal_type if it's categorical
    if 'meal_type' in df.columns:
        df['meal_type'] = df['meal_type'].astype('category').cat.codes

    # Fill any remaining missing values
    df = df.fillna(0)
    threshold = df["production_cost_total"].quantile(threshold)
    df = df[df["production_cost_total"] <= threshold]
    return df