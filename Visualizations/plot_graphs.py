import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Simulations.history as history
from Simulations.algorithms import *
from Simulations.instances import *
from scipy.special import lambertw

# rlim = 10
#
# eta = np.linspace(0, rlim, 100*rlim + 1)
# y = [np.min([x, np.ceil((x + 1)/2)]) for x in eta]
#
# plt.plot(eta, y)
# plt.savefig("Plots/min-eta.png")
# plt.show()



######

# gamma = 1
# phi = 100
#
# prices_daily = 4 * [1, phi]
# demands = 4 * [0, 1]
#
# pred_opt = 4 * [1, 0]
# pred_ftp = 4 * [0, 1]
#
# opt = FtP(0, 0, pred_opt)
# ftp = FtP(0, 0, pred_ftp)
#
# history.run_and_generate_history_df(phi, prices_daily, demands, [opt, ftp], False, gamma)

# #####################################################
#
# # plot competitive ratio of optimal online algorithm
#
# phi = np.linspace(1, 100, 1000)
# y_opt = 1/(lambertw((1-phi)/(np.exp(1)*phi)).real + 1)
# plt.plot(phi, y_opt)
#
# plt.show()
#
# #####################################################

# #  plot additive term in cost bound of FtP
#
# phi = np.linspace(1, 100, 1000)
# y_0 = [np.min([0, (x-1)*(-1) + x]) for x in phi]
# y_1 = [np.min([1*x, (x-1)*(1-1) + x]) for x in phi]
# y_2 = [np.min([2*x, (x-1)*(2-1) + x]) for x in phi]
#
# plt.plot(phi, y_0, label="$ \eta = 0 $")
# plt.plot(phi, y_1, label="$ \eta = 1 $")
# plt.plot(phi, y_2, label="$ \eta = 2 $")
#
# plt.show()

eta = np.linspace(0, 8, 1000)
print(eta)
# stock level based: [np.min([phi*n, (phi-1)*(n-1) + phi]) for n in eta]
# y_1 = [np.min([1*n, (1-1)*(n-1) + 1]) for n in eta]
# y_2 = [np.min([2*n, (2-1)*(n-1) + 2]) for n in eta]
# y_3 = [np.min([3*n, (3-1)*(n-1) + 3]) for n in eta]
# y_10 = [np.min([10*n, (10-1)*(n-1) + 10]) for n in eta]
# y_30 = [np.min([30*n, (30-1)*(n-1) + 30]) for n in eta]
# y_50 = [np.min([50*n, (50-1)*(n-1) + 50]) for n in eta]
# y_100 = [np.min([100*n, (100-1)*(n-1) + 100]) for n in eta]

# purchase amount based: [np.min([phi*n, phi + ((n-1)/2)*(phi-1)]) for n in eta]
y_1 = [np.min([1*n, 1 + ((n-1)/2)*(1-1)]) for n in eta]
y_2 = [np.min([2*n, 2 + ((n-1)/2)*(2-1)]) for n in eta]
y_3 = [np.min([3*n, 3 + ((n-1)/2)*(3-1)]) for n in eta]
y_10 = [np.min([10*n, 10 + ((n-1)/2)*(10-1)]) for n in eta]
y_30 = [np.min([30*n, 30 + ((n-1)/2)*(30-1)]) for n in eta]
y_50 = [np.min([50*n, 50 + ((n-1)/2)*(50-1)]) for n in eta]
y_100 = [np.min([100*n, 100 + ((n-1)/2)*(100-1)]) for n in eta]

plt.plot(eta, y_1, label="$ \\varphi = 1 $")
plt.plot(eta, y_2, label="$ \\varphi = 2 $")
plt.plot(eta, y_3, label="$ \\varphi = 3 $")
plt.plot(eta, y_10, label="$ \\varphi = 10 $")
# plt.plot(eta, y_30, label="$ \\varphi = 30 $")
plt.plot(eta, y_50, label="$ \\varphi = 50 $")
# plt.plot(eta, y_100, label="$ \\varphi = 100 $")

plt.xlim([0, 7])
plt.ylim([0, 15])

plt.legend()
plt.xlabel("$ \eta $")
plt.ylabel("$ \min\{ \\varphi\eta, (\\varphi-1)(\eta-1) + \\varphi \} $")

plt.show()