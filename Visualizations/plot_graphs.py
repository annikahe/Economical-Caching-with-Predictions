import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Simulations.history as history
from Simulations.algorithms import *
from Simulations.instances import *

# rlim = 10
#
# eta = np.linspace(0, rlim, 100*rlim + 1)
# y = [np.min([x, np.ceil((x + 1)/2)]) for x in eta]
#
# plt.plot(eta, y)
# plt.savefig("Plots/min-eta.png")
# plt.show()



######

gamma = 1
phi = 100

prices = 4 * [1, phi]
demands = 4 * [0, 1]

pred_opt = 4 * [1, 0]
pred_ftp = 4 * [0, 1]

opt = FtP(0, 0, pred_opt)
ftp = FtP(0, 0, pred_ftp)

history.run_and_generate_history_df(phi, prices, demands, [opt, ftp], False, gamma)