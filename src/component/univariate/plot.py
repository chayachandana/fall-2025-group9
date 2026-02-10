import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, model_name, plot_path):
    """
    Plots actual vs. predicted values and saves the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Actual Production Cost")
    plt.ylabel("Predicted Production Cost")
    plt.title(f"{model_name} Predicted vs Actual")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def plot_train_test_forecast(
    dates,
    values,
    split_idx,
    forecast_dates,
    forecast_values,
    title,
    plot_path,
):
    """
      plot of train, test, and forecast for a single (school_name, meal_type) series.
    - dates: all historic dates for that school+meal
    - values: all historic production_cost_total values
    - split_idx: index where train ends and test starts (same logic as training)
    - forecast_dates: dates of future forecasts
    - forecast_values: forecasted production_cost_total
    """
    dates = np.asarray(dates)
    values = np.asarray(values)
    forecast_dates = np.asarray(forecast_dates)
    forecast_values = np.asarray(forecast_values)

    train_dates = dates[:split_idx]
    train_vals  = values[:split_idx]

    test_dates = dates[split_idx:]
    test_vals  = values[split_idx:]

    plt.figure(figsize=(10, 5))
    plt.plot(train_dates, train_vals, label="Train", linewidth=1.5)
    plt.plot(test_dates, test_vals, label="Test", linewidth=1.5)
    plt.plot(forecast_dates, forecast_values, label="Forecast", linestyle="--", linewidth=1.5)

    plt.xlabel("Date")
    plt.ylabel("Production Cost Total")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()