
import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, out_dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.unsqueeze(1)
