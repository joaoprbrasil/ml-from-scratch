import numpy as np
import linear_models

X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([xi * 5 for xi in X])

w0, w1 = linear_models.linear_regression_closed_form(X, Y)

assert np.isclose(w0, 0.0)
assert np.isclose(w1, 5.0)
