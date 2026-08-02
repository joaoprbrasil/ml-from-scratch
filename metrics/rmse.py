import numpy as np

def root_mean_squared_error(Y: np.array, hat_Y: np.array) -> float:
    assert Y.shape == hat_Y.shape
    return np.sqrt(np.square(Y - hat_Y).sum() / len(Y))



