from algorithms import *
from instances import *
import offline as off
import numpy as np
import matplotlib.pyplot as plt
import history

def run_constant_prices_and_demands(c, num_rounds, gamma, phi, num_algs):

    epsilon = 1 / 100
    # phi = np.ceil(6/epsilon + 1)

    prices = num_rounds * [c]
    demands = num_rounds * [1]
    pred_A0 = num_rounds * [0]
    pred_A1 = num_rounds * [1]

    A0 = FtP(0, 0, pred_A0)
    A1 = FtP(0, 0, pred_A1)

    history.run_and_print_history(gamma, phi, prices, demands, [A0, A1], True)