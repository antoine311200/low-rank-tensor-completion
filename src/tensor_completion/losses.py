import numpy as np

def MAPE_loss(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE_loss(y_pred, y_true):
    return np.sqrt(np.mean((y_true - y_pred)**2))