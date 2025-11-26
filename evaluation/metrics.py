
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mase(y_true, y_pred):
    naive = np.mean(np.abs(np.diff(y_true)))
    return np.mean(np.abs(y_true - y_pred)) / (naive + 1e-8)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true)<1e-6, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred)/denom)) * 100.0

def basic_eval(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'mase': mase(y_true, y_pred)
    }
