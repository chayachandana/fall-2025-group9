import torch
import torch.nn as nn

class ForecastingModel(nn.Module):
    def __init__(self, model_type: str = "LSTM", input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 1, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        m = str(model_type).upper()
        if m == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout
            )
        elif m == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout
            )
        else:
            raise ValueError("MODEL_TYPE must be 'GRU' or 'LSTM'")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, window, 1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]      # (batch, hidden_dim)
        return self.fc(last)
    

class FeedForwardRegressor(nn.Module):
    """Simple MLP baseline for windowed regression."""
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (batch, in_dim)
        return self.net(x)
