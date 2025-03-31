import numpy as np

def mrrmse(y_true, y_pred):
    rmse_per_sample = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    return np.mean(rmse_per_sample)
