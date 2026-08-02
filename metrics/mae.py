import numpy as np

def mean_absolute_error(Y: np.array, hat_Y: np.array) -> float:
    assert Y.shape == hat_Y.shape
    return np.abs(Y - hat_Y).sum() / len(Y)

# %%
Y = np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
hat_Y = np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

assert mean_absolute_error(Y, hat_Y) == 0

Y = np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
hat_Y = np.array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

Y = np.array([2.16096008e+05, 2.11757560e+06, 5.08631331e+05, 2.16287313e+08,
       1.33624185e+06, 2.20888960e+06, 2.62248827e+06, 2.57414556e+05,
       9.50326613e+03, 2.16286073e+07])
hat_Y = np.array([2.16096008e+05, 2.11757560e+06, 5.08631331e+05, 2.16287313e+08,
       1.33624185e+06, 2.20888960e+06, 2.62248827e+06, 2.57414556e+05,
       9.50326613e+03, 2.16286073e+07])
assert mean_absolute_error(Y, hat_Y) == 0

Y = np.array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 100])
hat_Y = np.array([-11.,   0.,  11.,  22.,  33.,  44.,  55.,  66.,  77.,  88.])
assert mean_absolute_error(Y, hat_Y) == 6.4


