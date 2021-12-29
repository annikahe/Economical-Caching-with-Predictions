from decimal import *
import numpy as np
from numpy.random import default_rng


# Generating prices

def prices_uniform(length, phi):
    rng = default_rng()
    return [rng.uniform(1, phi) for _ in range(length)]


def prices_normal(length, phi, mu=1, sigma=0.1):
    rng = default_rng()
    return [np.clip(rng.normal(mu, sigma), 1, phi) for _ in range(length)]


# Generating demands

def demands_uniform(length):
    rng = default_rng()
    return [rng.uniform() for _ in range(length)]


def demands_normal(length, mu=0.5, sigma=0.1):
    rng = default_rng()
    return [np.max([0, rng.normal(mu, sigma)]) for _ in range(length)]


# Other

def example1(phi, num_repetitions=1):
    prices = 2 * num_repetitions * [1, phi]
    demands = 2 * num_repetitions * [1, 1]

    return prices, demands


def worst_case(phi, num_repetitions):
    prices = (2 * num_repetitions + 1) * [phi, 1, phi - 4, 1, 1]
    demands = (2 * num_repetitions + 1) * [1/phi, 1, 1, 1, 1]

    return prices, demands
