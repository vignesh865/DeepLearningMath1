import rmse
import numpy as np


def test_rmse():
    predictions = [1, 2, 3, 4, 5]
    targets = [1, 2, 3, 4, 5]

    assert np.all(rmse.rmse(predictions, targets) == 0)

    predictions = [2, 3, 4, 5, 6]
    targets = [1, 2, 3, 4, 5]

    assert np.all(rmse.rmse(predictions, targets) == 1)
