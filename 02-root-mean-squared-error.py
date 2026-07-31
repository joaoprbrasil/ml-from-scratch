import numpy as np

def root_mean_squared_error(Y: np.array, hat_Y: np.array) -> float:
    assert Y.shape == hat_Y.shape
    return np.sqrt(np.square(Y - hat_Y).sum() / len(Y))

# %%
Y = np.array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 100])
hat_Y = np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
result = root_mean_squared_error(Y, hat_Y)
assert np.isclose(result, 21.69792616818483)



