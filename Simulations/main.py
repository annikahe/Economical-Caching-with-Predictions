from algorithms import *
from instances import *
import offline as off
import numpy as np
import matplotlib.pyplot as plt
import history
import examples as ex


# num_repetitions = 2
#
# gamma = 1
# phi = 5
#
# prices_daily = (2 * num_repetitions + 1) * [phi, 1, phi - 4, 1, 1]
# demands = (2 * num_repetitions + 1) * [1/phi, 1, 1, 1, 1]
# pred_A0 = 5 * [0] + num_repetitions * [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# pred_A1 = [0, 1, 0, 1, 1] + num_repetitions * [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
# pred_A2 = [0, 1, 0, 1, 0] + (2*num_repetitions) * [1, 0, 1, 0, 0]
#
# Ex3A0 = FtP(0, 0, pred_A0)
# Ex3A1 = FtP(0, 0, pred_A1)
# Ex3A2 = FtP(0, 0, pred_A2)
#
# history.run_and_plot_history(phi, prices_daily, demands, [Ex3A0, Ex3A1, Ex3A2], True, gamma)

# gamma = 1
# phi = 10
#
# len_input = 10
#
# prices_daily = list((phi - 1) * np.random.rand(len_input) + 1)
# demands = list(np.random.rand(len_input))
# pred_A0 = list(np.random.rand(len_input))
# # pred_A1 = list(np.random.rand(len_input))
# # pred_A0 = len_input * [0]
# pred_A1 = len_input * [1]
# pred_A2 = len_input * [0]
#
# Ex4A0 = FtP(0, 0, pred_A0)
# Ex4A1 = FtP(0, 0, pred_A1)
# Ex4A2 = FtP(0, 0, pred_A2)
#
# history.run_and_plot_history(phi, prices_daily, demands, [Ex4A0, Ex4A1, Ex4A2], True, gamma)

num_repetitions = 6

gamma = 1
phi = 5

prices = 2 * num_repetitions * [1, phi]
demands = 2 * num_repetitions * [1, 1]
pred_A0 = 4 * num_repetitions * [0]
pred_A1 = (num_repetitions - 3) * [1, 0] + (num_repetitions + 3) * [0, 1]
pred_A2 = [0, 1] + (num_repetitions - 2) * [1, 1] + (num_repetitions + 1) * [1, 0]

Ex5A0 = FtP(0, 0, pred_A0)
Ex5A1 = FtP(0, 0, pred_A1)
Ex5A2 = FtP(0, 0, pred_A2)

history.run_and_plot_history(phi, prices, demands, [Ex5A0, Ex5A1, Ex5A2], False, gamma, "Ex5_2")
# history.run_and_plot_history_mindet_sections(phi, prices, demands, [Ex5A0, Ex5A1, Ex5A2], True, gamma, "Ex5_2")
