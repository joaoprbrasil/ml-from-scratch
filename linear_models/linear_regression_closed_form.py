import numpy as np

def linear_regression_closed_form(X: np.array, Y: np.array) -> (float, float):
    bar_x = X.mean()
    bar_y = Y.mean()

    # dividend = sum(np.array([xi - bar_x for xi in X]) *
    #                np.array([yi - bar_y for yi in Y]))

    dividend = np.sum((X - bar_x) * (Y - bar_y))

    divisor = np.sum((xi - bar_x)**2 for xi in X)

    weight1 = dividend / divisor

    weight0 = bar_y - (weight1 * bar_x)

    return weight0, weight1


