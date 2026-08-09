import numpy as np

def linear_regression_closed_form(X: np.array, Y: np.array) -> (float, float):
    bar_x = X.mean()
    bar_y = Y.mean()

    covariance = np.sum((X - bar_x) * (Y - bar_y))

    variance = np.sum((xi - bar_x)**2 for xi in X)

    weight1 = covariance / variance

    weight0 = bar_y - (weight1 * bar_x)

    return weight0, weight1


