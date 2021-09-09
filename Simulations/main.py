from algorithms import *
from instances import *
import offline as off
import numpy as np
import matplotlib.pyplot as plt
import history
import examples as ex


num_repetitions = 100

gamma = 1
phi = 5

prices = (2 * num_repetitions + 1) * [phi, 1, phi - 4, 1, 1]
demands = (2 * num_repetitions + 1) * [1/phi, 1, 1, 1, 1]
pred_A0 = 5 * [0] + num_repetitions * [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
pred_A1 = [0, 1, 0, 1, 1] + num_repetitions * [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

Ex3A0 = FtP(0, 0, pred_A0)
Ex3A1 = FtP(0, 0, pred_A1)

history.run_and_print_history_latex(phi, prices, demands, [Ex3A0, Ex3A1], True, gamma)
