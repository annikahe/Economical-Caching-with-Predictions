import numpy as np
import matplotlib.pyplot as plt


def h(i, x, prices, demands, phi):
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
    sorted_prices = [0] + sorted(prices)
    c_int = 0
    for j in range(i):
        y = np.max([h(i - 1, sorted_prices[j+1], prices, demands, phi) + demands[i - 1] - 1, 0])
        c_int += (sorted_prices[j+1] - sorted_prices[j]) * y
    return c_int


def cost_opt(prices, demands, phi):
    c_sum = 0
    for i in range(len(prices)):
        c_sum += c_integral(i + 1, prices, demands, phi)
    return c_sum

