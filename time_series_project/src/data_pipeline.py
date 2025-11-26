
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(values):
    scaler = MinMaxScaler()
    return scaler.fit_transform(values), scaler

def create_dataset(data, seq_len):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def train_test_split(X, y, split=0.8):
    idx=int(len(X)*split)
    return X[:idx], X[idx:], y[:idx], y[idx:]
