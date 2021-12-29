from Simulations.history import *
import Simulations.pickle_helpers as ph


# Example used in talk

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

mindet = MinDetHistory(gamma, phi, prices, demands, [Ex5A0, Ex5A1, Ex5A2])
mindet.run_full()
mindet.plot_history_mindet_sections()

opt_pred = pred.opt_off(prices, demands)
opt_off = History(gamma, phi, prices, demands, FtP(0, 0, opt_pred))
opt_off.run_full()

ph.save_objects([mindet, opt_off], 'Instances/ex_talk.pkl')
