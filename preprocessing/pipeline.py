
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_series(series):
    scaler = MinMaxScaler()
    arr = series.reshape(-1,1)
    scaled = scaler.fit_transform(arr).squeeze(-1)
    return scaled, scaler

def create_sequences(series, seq_len, pred_len=1):
    X, y = [], []
    for i in range(len(series)-seq_len-pred_len+1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
