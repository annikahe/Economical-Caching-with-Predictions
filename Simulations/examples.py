from Simulations.history import *


def run_constant_prices_and_demands(c, num_rounds, gamma, phi, num_algs):

    epsilon = 1 / 100
    # phi = np.ceil(6/epsilon + 1)

    prices = num_rounds * [c]
    demands = num_rounds * [1]
    pred_A0 = num_rounds * [0]
    pred_A1 = num_rounds * [1]

    A0 = FtP(0, 0, pred_A0)
    A1 = FtP(0, 0, pred_A1)

    mindet = MinDetHistory(gamma, phi, prices, demands, [A0, A1])

    mindet.run_full()
    mindet.print_history_table()

    # history.run_and_print_history_table(phi, prices, demands, [A0, A1], True, gamma)