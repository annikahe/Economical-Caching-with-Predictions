import numpy as np
from numpy.random import default_rng
import Simulations.offline as off


# Generate predictions independently from the instance and the optimal solution

def predictions_uniform(length):
    rng = default_rng()
    return [rng.uniform() for _ in range(length)]


def predictions_normal(length, mu=0.5, sigma=0.1):
    rng = default_rng()
    return [np.clip(rng.normal(mu, sigma), 0, 1) for _ in range(length)]


def predictions_0(length):
    return [0] * len(length)


def predictions_1(length):
    return [1] * len(length)


# generate optimal predictions

def opt_off(prices, demands):
    return off.opt_stock_levels(prices, demands)


# Generate predictions based on the optimal offline solution

def predictions_normal_off(off_solution, sigma=0.1):
    rng = default_rng()
    return [np.clip(rng.normal(x, sigma), 0, 1) for x in off_solution]


# Other

def pred_example1(num_repetitions):
    pred_A0 = 4 * num_repetitions * [0]
    pred_A1 = (num_repetitions - 3) * [1, 0] + (num_repetitions + 3) * [0, 1]
    pred_A2 = [0, 1] + (num_repetitions - 2) * [1, 1] + (num_repetitions + 1) * [1, 0]
    # Todo: return


def pred_worst_case(num_repetitions):
    pred_A0 = 5 * [0] + num_repetitions * [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    pred_A1 = [0, 1, 0, 1, 1] + num_repetitions * [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    pred_A2 = [0, 1, 0, 1, 0] + (2*num_repetitions) * [1, 0, 1, 0, 0]
    # Todo: return
