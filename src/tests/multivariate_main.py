import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import warnings
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.component.multivariate.utils import preprocessing
from src.component.multivariate.training import train_linear_regression, train_xgboost, train_fnn, train_gru

warnings.filterwarnings("ignore")
torch.manual_seed(42)

def load_and_preprocess_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(current_dir, '..', 'Data', 'Output', 'meals_combined.csv')
    df = pd.read_csv(CSV_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df = df.sort_values(['school_name', 'meal_type', 'date'])
    preprocessed_data = preprocessing(df)
    if "production_cost_total" in preprocessed_data.columns:
        preprocessed_data["production_cost_total"] = preprocessed_data["production_cost_total"].replace(
            {"\$": "", ",": ""}, regex=True
        ).astype(float)
    return preprocessed_data

def main():
    preprocessed_data = load_and_preprocess_data()
    features = ["served_total", "planned_total", "discarded_total", "left_over_total"]
    target = "production_cost_total"
    X = preprocessed_data[features].values
    y = preprocessed_data[target].values
    X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(X, y, test_size=0.2, random_state=42)

    train_linear_regression(X_train_tab, y_train_tab, X_test_tab, y_test_tab)
    train_xgboost(X_train_tab, y_train_tab, X_test_tab, y_test_tab)
    train_fnn(X_train_tab, y_train_tab, X_test_tab, y_test_tab, features)
    train_gru(preprocessed_data, features, target)

if __name__ == "__main__":
    main()
