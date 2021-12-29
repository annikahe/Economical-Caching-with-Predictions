from Simulations.algorithms import *
import matplotlib.pyplot as plt
import Simulations.history as history


# num_repetitions = 2
#
# gamma = 1
# phi = 5
#
# prices = (2 * num_repetitions + 1) * [phi, 1, phi - 4, 1, 1]
# demands = (2 * num_repetitions + 1) * [1/phi, 1, 1, 1, 1]
# pred_A0 = 5 * [0] + num_repetitions * [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# pred_A1 = [0, 1, 0, 1, 1] + num_repetitions * [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
# pred_A2 = [0, 1, 0, 1, 0] + (2*num_repetitions) * [1, 0, 1, 0, 0]
#
# Ex3A0 = FtP(0, 0, pred_A0)
# Ex3A1 = FtP(0, 0, pred_A1)
# Ex3A2 = FtP(0, 0, pred_A2)
#
# history.run_and_plot_history(phi, prices, demands, [Ex3A0, Ex3A1, Ex3A2], True, gamma)

# gamma = 1
# phi = 10
#
# len_input = 10
#
# prices = list((phi - 1) * np.random.rand(len_input) + 1)
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
# history.run_and_plot_history(phi, prices, demands, [Ex4A0, Ex4A1, Ex4A2], True, gamma)

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

# HistEx5A0 = history.History(gamma, phi, prices, demands, Ex5A0)
# HistEx5A1 = history.History(gamma, phi, prices, demands, Ex5A1)
# HistEx5A2 = history.History(gamma, phi, prices, demands, Ex5A2)
#
# HistEx5A0.run_full()
# HistEx5A1.run_full()
# HistEx5A2.run_full()
#
# HistEx5A0.plot_history_costs("A_0", "red")
# HistEx5A1.plot_history_costs("A_1", "blue")
# HistEx5A2.plot_history_costs("A_2", "green")

mindet = history.MinDetHistory(gamma, phi, prices, demands, [Ex5A0, Ex5A1, Ex5A2])
mindet.run_full()
mindet.plot_history_mindet_sections()

plt.show()
# history.run_and_plot_history_mindet_sections(phi, prices, demands, [Ex5A0, Ex5A1, Ex5A2], True, gamma, "Ex5_2")
