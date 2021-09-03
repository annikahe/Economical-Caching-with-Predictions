from algorithms import *
from instances import *
import offline as off
import numpy as np
import matplotlib.pyplot as plt
import history
import pandas as pd
import examples as ex


# gamma = 1
# phi = 5
#
# num_repetitions = 5
#
# prices = num_repetitions * [1, phi]
# demands = num_repetitions * [0, 1]
#
# opt = num_repetitions * [1, 0]
# worst = num_repetitions * [0, 1]

# a0 = AlgorithmEx1(0, 0)
# a1 = AlgorithmEx1(0, 0)
# a2 = RPA(0, 0)
# a3 = FtP(0, 0, opt)
# a4 = FtP(0, 0, worst)
# mindet = MinDet(0, 0, [a0, a1])

# a5 = AlgorithmEx2(0, 0, opt)
# a6 = AlgorithmEx2(0, 0, opt)


# for i in range(len(prices)):
#     #mindet.run(gamma, phi, prices[i], demands[i])
#     #print(mindet.alg_list[0].cost)
#     a3.run(gamma, phi, prices[i], demands[i])
#     print(a3.cost)
#     a4.run(gamma, phi, prices[i], demands[i])
#     print(a4.cost)
#     #print(mindet.current_alg)
#     #print(mindet.x)
#
# # print(f"A0 turn: {a0.turn}")
# # print(f"A1 turn: {a1.turn}")
# # print(mindet.x)
# # print(mindet.cost)
# # print(mindet.stock)
# # a0.buy(1,1)
# # print(a0.x)
#
# print(off.cost_opt(prices, demands, phi))

# x = np.linspace(0, phi, 500)
# hx = []
# for j in range(len(x)):
#     hx.append(off.h(4, x[j], prices, demands, phi))
# plt.plot(x, hx)
# plt.ylim(0,1.1)
# plt.show()

#history.run_and_generate_history_df(gamma, phi, prices, demands, [a0, a1], True)

#history.run_and_print_history(gamma, phi, prices, demands, [a5, a6], True)


def to_latex_string(input):
    return "$ " + str(input) + " $"


tls = to_latex_string


def lin_vphi(mult, add):
    vphi = "\\varphi"
    if add >= 0:
        op = "+"
    else:
        op = ""

    latex_string = ""

    if mult != 0:
        if mult != 1:
            latex_string += f"{mult}"
        latex_string += f"{vphi}"

    if add != 0:
        if mult != 0:
            latex_string += f"{op}"
        latex_string += f"{add}"

    return to_latex_string(latex_string)

lv = lin_vphi

# df = pd.DataFrame([[1,  lv(1, -1),          1,          1,      lv(1, -2),      1,      lv(1, -3),      1],
#                    [0,          1,          1,          0,              1,      1,              1,      1],
#                    [0,          1,          1,          1,              0,      2,              0,      1],
#                    [0,  lv(1, -1),   lv(1, 0),   lv(1, 1), lv(1, 1), lv(1, 3), lv(1, 3), ],
#                    [0,                0, 0, 1,                       0, 1,                  0, 1],
#                    [],
#                    [],
#                    [],
#                    [],
#                    [],
#                    []],
#                   index=["Price", "Demand", f"$ x(A_0) $", f"$ \cost(A_0) $", "$ \stock(A_0) $",
#                          f"$ x(A_1) $", f"$ \cost(A_1) $", "$ \stock(A_1) $",
#                          "$ x(\mindet) $", "$ \cost(\mindet) $", "Alg. currently exec. by $ \mindet $"])
#
# print(df)


# print(lin_vphi(2,2))
# print(lv(0,3))
# print(lv(1,100))
# print(lv(1, 0))
#
# num_repetitions = 100
#
# phi = 100
# gamma = 1
#
# prices = 2 * num_repetitions * [1, phi-2, 1] + [1, phi-2, 1]
# demands = 2 * num_repetitions * [1, 1, 1] + [1, 1, 1]
# pred_A_0 = [0, 0, 0] + num_repetitions * [1, 0, 1, 0, 1, 0]
# pred_A_1 = [1, 0, 1] + num_repetitions * [0, 1, 0, 1, 0, 1]
#
# A0 = FtP(0, 0, pred_A_0)
# A1 = FtP(0, 0, pred_A_1)
#
# history.run_and_print_history(gamma, phi, prices, demands, [A0, A1], True)

##########

num_repetitions = 100

gamma = 1

epsilon = 1/100
phi = 5
#phi = np.ceil(6/epsilon + 1)

prices = (2 * num_repetitions + 1) * [phi, 1, phi - 4, 1, 1]
demands = (2 * num_repetitions + 1) * [1/phi, 1, 1, 1, 1]
pred_A0 = 5 * [0] + num_repetitions * [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
pred_A1 = [0, 1, 0, 1, 1] + num_repetitions * [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

Ex3A0 = FtP(0, 0, pred_A0)
Ex3A1 = FtP(0, 0, pred_A1)

history.run_and_print_history(gamma, phi, prices, demands, [Ex3A0, Ex3A1], True)
#history.run_and_generate_history_df(gamma, phi, prices, demands, [Ex3A0, Ex3A1], True)

#######

epsilon = 1/10
phi = np.ceil(6/epsilon + 1)

k = np.ceil((-6*epsilon - phi + 18)/(2*epsilon*phi - 12))

print(f"k = {k}, phi = {phi}")


ex.run_constant_prices_and_demands(1, 100, 1, 2, 2)

###

gamma = 1
phi = 100

prices = 4 * [1, phi]
demands = 4 * [0, 1]

pred_opt = 4 * [1, 0]
pred_ftp = 4 * [0, 1]

opt = FtP(0, 0, pred_opt)
ftp = FtP(0, 0, pred_ftp)

history.run_and_generate_history_df(gamma, phi, prices, demands, [opt, ftp], False)

# gamma = 1
# phi = 100
#
# n_iterations = 5
#
# #prices = n_iterations * [1, phi]
# #demands = n_iterations * [0, 1]
#
# prices = (phi - 1) * np.random.random(n_iterations) + 1
# demands = np.random.random(n_iterations)
#
#
# opt = 10 * [1, 0]
# worst = 10 * [0, 1]
#
# a0 = AlgorithmRandom(0, 0)
# a1 = AlgorithmRandom(0, 0)
# a2 = AlgorithmRandom(0, 0)
#
# algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks = run_and_generate_history(gamma, phi, prices, demands, [a0, a1, a2], True)
# print(algs_purchases)
# print(mindet_purchases)
# print(mindet_current_algs)
# #print(get_breakpoints(mindet_current_algs))
# #print(get_lengths_of_chunks(mindet_current_algs))
# print(get_ends_of_chunks(mindet_current_algs))
#
# #plot_history(algs_purchases, mindet_purchases, mindet_current_algs)
#
# #run_and_print_history(gamma, phi, prices, demands, [a0, a1, a2], True)
#
# run_and_generate_history_df(gamma, phi, prices, demands, [a0, a1, a2], True)

# ### Test optimal offline algorithm
# phi = 100
#
# prices = [0.5 * phi, 0.75 * phi, phi]
# demands = [1, 0.5, 0.25]
#
# print(h(3, 0.6 * phi, prices, demands, phi))
#
# x = np.linspace(0, phi, 500)
# hx = []
# for j in range(len(x)):
#     hx.append(h(3, x[j], prices, demands, phi))
# plt.plot(x, hx)
# plt.ylim(0,1.1)
# plt.show()
#
# c_opt = cost_opt(prices, demands, phi)
# print(c_opt)

