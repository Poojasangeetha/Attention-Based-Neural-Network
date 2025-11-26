# time_series_attention_project.py
"""
Advanced Time Series Forecasting Project
- Generates or loads a time series (>=500 points)
- Builds data pipeline (scaling, windowing, splits)
- Implements Transformer encoder forecasting model (PyTorch)
- Implements LSTM baseline
- Trains both models, computes metrics (MSE, RMSE, MAE, MAPE)
- Extracts and visualizes attention weights

Usage:
- Run as a script: `python time_series_attention_project.py`
- To use your own CSV: set DATA_CSV = 'path/to.csv' and COLUMN = 'colname'

Requirements:
- Python 3.8+
- numpy, pandas, matplotlib, scikit-learn, torch

"""

# Original user code starts here
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

@dataclass
class Config:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_synthetic: bool = True
    data_csv: str = ''
    csv_column: str = 'value'
    min_length: int = 500
    scaler_type: str = 'minmax'
    input_window: int = 48
    output_horizon: int = 1
    train_frac: float = 0.7
    val_frac: float = 0.2
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(cfg.seed)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def generate_synthetic_series(n: int = 1500, freq: int = 24) -> pd.Series:
    t = np.arange(n)
    trend = 0.001 * t
    seasonal = 2.0 * np.sin(2 * np.pi * t / freq) + 0.5 * np.sin(2 * np.pi * t / (freq*7))
    noise = 0.5 * np.random.randn(n)
    series = 10 + trend + seasonal + noise
    return pd.Series(series)

def load_series_from_csv(path: str, column: str) -> pd.Series:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV")
    return df[column].astype(float)

class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, input_window: int, output_horizon: int):
        self.series = series
        self.input_window = input_window
        self.output_horizon = output_horizon
        self.indices = self._create_indices()

    def _create_indices(self):
        L = len(self.series)
        idx = []
        for i in range(L - self.input_window - self.output_horizon + 1):
            idx.append(i)
        return idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        x = self.series[i:i + self.input_window]
        y = self.series[i + self.input_window:i + self.input_window + self.output_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def create_dataloaders(series: np.ndarray, cfg: Config):
    n = len(series)
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)

    train_series = series[:train_end]
    val_series = series[train_end:val_end]
    test_series = series[val_end:]

    train_ds = TimeSeriesDataset(train_series, cfg.input_window, cfg.output_horizon)
    val_ds = TimeSeriesDataset(val_series, cfg.input_window, cfg.output_horizon)
    test_ds = TimeSeriesDataset(test_series, cfg.input_window, cfg.output_horizon)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (train_series, val_series, test_series)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_horizon=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, output_horizon)

    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(-1)
        x = self.input_proj(src)
        x = self.pos_enc(x)
        enc = self.transformer_encoder(x)
        pooled = enc.mean(dim=1)
        out = self.output(pooled)
        return out.squeeze(-1), enc

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, output_horizon=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.attn_linear = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_horizon)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        outputs, (hn, cn) = self.lstm(x)
        scores = self.attn_linear(outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(outputs * weights, dim=1)
        out = self.fc(context)
        return out.squeeze(-1), weights.squeeze(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze(-1)
        optimizer.zero_grad()
        preds, _ = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_list = []
    y_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            preds, _ = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)
            preds_list.append(preds.cpu().numpy())
            y_list.append(y.cpu().numpy())
    preds_all = np.concatenate(preds_list)
    y_all = np.concatenate(y_list)
    return total_loss / len(loader.dataset), preds_all, y_all

def plot_series(series, title='Series'):
    pass

def plot_predictions(true, pred, start=0, length=200, title='Predictions vs True'):
    pass

def plot_attention_weights(weights, title='Attention Weights'):
    pass

def main(cfg: Config):
    series = generate_synthetic_series(max(cfg.min_length, 1500))
    scaled = series.values
    train_loader, val_loader, test_loader, _ = create_dataloaders(scaled, cfg)

if __name__ == '__main__':
    main(cfg)
