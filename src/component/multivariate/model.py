#%%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  

class ForecastingModel(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(ForecastingModel, self).__init__()
        self.model_type = model_type.upper()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError("model_type must be 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):

        if lengths is not None:

            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_input)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)

            last_outputs = torch.stack([out[i, l - 1, :] for i, l in enumerate(lengths)])
        else:
            out, _ = self.rnn(x)
            last_outputs = out[:, -1, :]

        return self.fc(last_outputs)
    
class FeedForwardNN(nn.Module):
    """
    Simple feed-forward neural network for tabular regression.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(FeedForwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)   
