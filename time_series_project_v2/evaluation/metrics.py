
import numpy as np

def mase(y_true, y_pred):
    naive = np.abs(np.diff(y_true)).mean()
    return np.mean(np.abs(y_true - y_pred)) / naive

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).mean())

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
