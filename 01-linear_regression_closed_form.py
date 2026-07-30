import numpy as np

def simple_linear_regression_closed_form(X: np.array, Y: np.array) -> (float, float):
    bar_x = X.mean()
    bar_y = Y.mean()

    # dividend = sum(np.array([xi - bar_x for xi in X]) *
    #                np.array([yi - bar_y for yi in Y]))

    dividend = np.sum((X - bar_x) * (Y - bar_y))

    divisor = np.sum((xi - bar_x)**2 for xi in X)

    weight1 = dividend / divisor

    weight0 = bar_y - (weight1 * bar_x)

    return weight0, weight1

# %%
X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([xi * 5 for xi in X])

w0, w1 = simple_linear_regression_closed_form(X, Y)

assert np.isclose(w0, 0.0)
assert np.isclose(w1, 5.0)