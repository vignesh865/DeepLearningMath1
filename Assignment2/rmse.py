import numpy as np


def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)

    rmse = np.sqrt(np.sum((tar - pred) ** 2) / tar.shape[0])  # TODO: Implement RMSE Calculation here...
    return rmse
