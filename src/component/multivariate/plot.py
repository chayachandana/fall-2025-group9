import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred, model_name, plot_path):
    """
    Plots actual vs. predicted values and saves the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction line")
    plt.xlabel("Actual Production Cost")
    plt.ylabel(f"Predicted Production Cost ({model_name})")
    plt.title(f"{model_name}: Predicted vs Actual")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
