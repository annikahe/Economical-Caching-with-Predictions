import numpy as np
import matplotlib.pyplot as plt


def h(i, x, prices, demands, phi):
    """
    We are looking for a solution that covers the requests up to time step i and has a full stock at the end of step i.
    This function describes at what prices the stock, which is present at the end of time step i, was purchased.
    The term 1 - h(i, x, prices, demands, phi) indicates the fraction of the stock
    that was bought at a price of x or better.

    :param i: integer
        i in {0, ..., T}
    :param x: float
        some price x in [1, phi]
    :param prices: list
        complete list of all input prices
    :param demands: list
        complete list of all input demands
    :param phi: float
        upper bound on the prices, phi >= 1
    :return: float
        evaluation of the function h_i at x.
    """
    if i == 0:
        if x == phi:
            return 0
        else:
            return 1
    else:
        if x <= prices[i-1]:
            return np.min([h(i - 1, x, prices, demands, phi) + demands[i - 1], 1])
        else:
            return 0


def c_integral(i, prices, demands, phi):
    """
    Computes the cost that the optimal offline algorithm incurs to cover the demand v_i in time step i.

    :param i: integer
        time step, i in {0, ..., T}
    :param prices: list
        complete list of all input prices
    :param demands: list
        complete list of all input demands
    :param phi: float
        upper bound on the prices, phi >= 1
    :return: float
        cost that OPT incurs to cover v_i
    """
    sorted_prices = [0] + sorted(prices)
    c_int = 0
    for j in range(i):
        y = np.max([h(i - 1, sorted_prices[j+1], prices, demands, phi) + demands[i - 1] - 1, 0])
        c_int += (sorted_prices[j+1] - sorted_prices[j]) * y
    return c_int


def cost_opt(prices, demands, phi):
    """
    Computes the total cost of the optimal offline algorithm for the Economical Caching Problem.

    :param prices: list
        Complete list of prices (input)
    :param demands: list
        Complete list of demands (input)
    :param phi: float
        Upper bound on the prices.
    :return: float
        Total cost of OPT incurred during the whole input sequence.
    """
    c_sum = 0
    for i in range(len(prices)):
        c_sum += c_integral(i + 1, prices, demands, phi)
    return c_sum


