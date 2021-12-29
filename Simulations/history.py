from Simulations.algorithms import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd

import numpy as np


""" Common Parameters:
    :param gamma:
    :param phi: float
        Upper bound on the prices.
    :param prices: List
        List of incoming prices.
    :param demands: List
        List of incoming demands.
"""

color_list = ["red", "blue", "green", "purple", "magenta", "grey"]


# def run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma=1):
#     """
#         Executes given algorithms on the complete input sequence.
#         Can run the algorithms independently from each other or as input for MIN^det.
#
#         :param gamma:
#         :param phi:
#         :param prices:
#         :param demands:
#         :param alg_list: list
#             List of algorithms for the Economical Caching Problem.
#         :param with_mindet: boolean
#             Decides whether to simulate the algorithms with or without MIN^det.
#             with_mindet = True: Generate an instance of MIN^det with the given algorithms as input.
#                                 Run this algorithm on the given input sequence and generate the history.
#             with_mindet = False: Only run the given algorithms (independently from each other)
#                                 and generate the history.
#         :return:
#             - numpy array of arrays: inner arrays contain the purchase amounts of all input algorithms
#                                      in the individual time steps.
#             - numpy array of arrays: inner arrays contain the costs the different input algorithms
#                                      accumulated until the individual time steps.
#             - numpy array of arrays: inner arrays contain the stock levels of the different input algorithms
#                                      in the individual time steps.
#         """
#     algs_purchases = [[] for i in range(len(alg_list))]
#     algs_acc_costs = [[] for i in range(len(alg_list))]
#     algs_stocks = [[] for i in range(len(alg_list))]
#
#     if with_mindet:
#         mindet = MinDet(0, 0, alg_list)
#
#     mindet_purchases = []
#     mindet_current_algs = []
#     mindet_acc_costs = []
#     mindet_stocks = []
#     mindet_cycles = []
#
#     for i in range(len(prices)):
#         price = prices[i]
#         demand = demands[i]
#         if with_mindet:
#             mindet.run(gamma, phi, price, demand)
#             mindet_current_algs.append(mindet.current_alg)
#         for j, alg in enumerate(alg_list):
#             if not with_mindet:
#                 alg.run(gamma, phi, price, demand)
#             algs_purchases[j].append(alg.x)
#             algs_acc_costs[j].append(alg.cost)
#             algs_stocks[j].append(alg.stock)
#         if with_mindet:
#             mindet_purchases.append(mindet.x)
#             mindet_acc_costs.append(mindet.cost)
#             mindet_stocks.append(mindet.stock)
#             mindet_cycles.append(mindet.cycle)
#
#     return np.array(algs_purchases), np.array(algs_acc_costs), np.array(algs_stocks), mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles


def remove_trailing_zeros(val):
    """
    Remove redundant trailing zeros after the decimal point from a float and output the result as a string.

    :param val: float
    :return: string
    """
    val = str(val)
    return val.rstrip('0').rstrip('.') if '.' in val else val


def color_cell_latex(val):
    """
    This function is aimed at coloring cells inside a latex table.
    A given content of a cell is extended by the latex command to color a cell in yellow.

    :param val: string or float
        Content of the latex table cell.
    :return: string
        Assemblage of the latex command to color a cell in yellow and the cell content.
    """
    return "\cellcolor{yellow!50} " + str(val)


# def run_and_generate_history_df(phi, prices, demands, alg_list, with_mindet, gamma=1):
#     """
#     Run the algorithms given in alg_list
#     :param gamma:
#     :param phi:
#     :param prices:
#     :param demands:
#     :param alg_list:
#     :param with_mindet:
#     :return:
#     """
#     algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks = \
#         run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)
#
#     len_input = len(prices)  # length of the input sequence
#     columns = list(range(len_input))
#
#     df = pd.concat([pd.DataFrame(columns=columns),
#                     pd.DataFrame([prices], columns=columns, index=["Price"]),
#                     pd.DataFrame([demands], columns=columns, index=["Demand"])])
#
#     for i in range(len(algs_purchases)):
#         df = df.append(pd.DataFrame([map(str, list(algs_purchases[i]))], columns=columns, index=[f"$ x(A_{i}) $"]))
#         df = df.append(pd.DataFrame([list(algs_acc_costs[i])], columns=columns, index=[f"$ \cost(A_{i}) $"]))
#         df = df.append(pd.DataFrame([list(algs_stocks[i])], columns=columns, index=[f"$ \stock(A_{i}) $"]))
#
#     if with_mindet:
#         df = df.append(pd.DataFrame([mindet_purchases], columns=columns, index=["$ x(\mindet) $"]))
#         df = df.append(pd.DataFrame([mindet_acc_costs], columns=columns, index=["$ \cost(\mindet) $"]))
#         df = df.append(pd.DataFrame([mindet_stocks], columns=columns, index=["$ \stock(\mindet) $"]))
#         df = df.append(pd.DataFrame([mindet_current_algs], columns=columns, index=["Alg. currently exec. by $ \mindet $"]))
#
#     df = df.applymap(lambda x: remove_trailing_zeros(x))
#
#     if with_mindet:
#         for i in range(len_input):
#             if int(df.at["Alg. currently exec. by $ \mindet $", i]) == 0:
#                 df.at["$ x(A_0) $", i] = color_cell_latex(df.at["$ x(A_0) $", i])
#             elif int(df.at["Alg. currently exec. by $ \mindet $", i]) == 1:
#                 df.at["$ x(A_1) $", i] = color_cell_latex(df.at["$ x(A_1) $", i])
#
#     return df


# def run_and_print_history_latex(phi, prices, demands, alg_list, with_mindet, gamma=1):
#     df = run_and_generate_history_df(phi, prices, demands, alg_list, with_mindet, gamma)
#     print(df.to_latex(index=True, escape=False))


# def run_and_print_history_table(phi, prices, demands, alg_list, with_mindet, gamma=1):
#     algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks \
#         = run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)
#
#     len_input = len(prices)  # length of the input sequence
#
#     output_table = PrettyTable([""] + list(range(len_input)))
#
#     output_table.add_row(["Price"] + list(prices))
#     output_table.add_row(["Demand"] + list(demands))
#
#     for i in range(len(alg_list)):
#         output_table.add_row([f"x(ALG_{i})"] + list(algs_purchases[i]))
#         output_table.add_row([f"cost(ALG_{i})"] + list(algs_acc_costs[i]))
#         output_table.add_row([f"stock(ALG_{i})"] + list(algs_stocks[i]))
#
#     if with_mindet:
#         output_table.add_row(["x(MIN^det)"] + list(mindet_purchases))
#         output_table.add_row(["cost(MIN^det)"] + list(mindet_acc_costs))
#         output_table.add_row(["Current_Alg"] + list(mindet_current_algs))
#         output_table.add_row(["stock(MIN^det)"] + list(mindet_stocks))
#         output_table.add_row(["Ratio cost(MIN^det)/min(cost(A_i))"] + [mindet_acc_costs[i]/min(algs_acc_costs[0][i], algs_acc_costs[1][i]) for i in range(len(prices))])
#
#     print(output_table)


# def plot_history(algs_quantity, with_color=True):
#     """
#     Plots the development of a certain quantity of the given algorithms.
#     :param algs_quantity: List
#         A list with the values of some quantity of the online algorithms (e.g. the accumulated cost in every step)..
#     :param with_color: Boolean
#         Determines whether to plot the quantities in different colors or not.
#     """
#     num_algs = len(algs_quantity)
#     len_input = len(algs_quantity[0])
#
#     if with_color:
#         for i in range(num_algs):
#             plt.plot(algs_quantity[i], '-o', color=color_list[i])
#             plt.annotate(f"$A_{i}$", (len_input - .2, algs_quantity[i][-1]), color=color_list[i])
#     else:
#         for i in range(num_algs):
#             plt.plot(algs_quantity[i], '-o', color="k")
#             plt.annotate(f"$A_{i}$", (len_input - .2, algs_quantity[i][-1]), color=color_list[i])
#
#     plt.xlabel("Time step $ t $")
#     plt.ylabel("Accumulated Costs $ cost_t $")
#
#     plt.xlim([0, len_input + 1])


# def plot_history_mindet_sections(algs_quantity, mindet_purchases, mindet_current_algs, mindet_cycles, phi, gamma, name):
#     num_algs = len(algs_quantity)
#
#     plot_history(algs_quantity, with_color=False)
#
#     ends_chunks = get_ends_of_chunks(mindet_cycles)
#
#     start = 0
#     for l in range(len(ends_chunks)):
#         end = ends_chunks[l]
#         current_alg = mindet_current_algs[end]
#         plt.plot(range(start, end+1), algs_quantity[current_alg][start:end + 1], '-o', color=color_list[current_alg], label=f"$ A_{current_alg} $")
#         start = end
#
#     # dotted horizontal lines
#     # for l in range(max(mindet_cycles)):
#     #     plt.hlines((l + 1) * gamma * phi, 0, len(mindet_current_algs), linestyle='dotted', color=color_list[l % num_algs], linewidth=0.5)
#
#     # solid horizontal lines
#     # for l in mindet_cycles:
#     #     plt.hlines((l + 1) * gamma * phi, 0, len(mindet_current_algs), color=color_list[l % num_algs], linewidth=0.8)
#
#     plt.xticks(ends_chunks)
#     plt.yticks([(l + 1) * gamma * phi for l in mindet_cycles])
#
#     # history
#     lines = []
#     for current_alg in range(num_algs):
#         line, = plt.plot(range(1), range(1), '-o', color=color_list[current_alg], label="MIN$^{det}$ follows" + f"$ A_{current_alg} $")
#         lines.append(line,)
#     labels = [f"follow $ A_{current_alg} $" for current_alg in range(num_algs)]
#     plt.legend(lines, labels)


# def run_and_plot_history(phi, prices, demands, alg_list, with_mindet, gamma, name):
#     algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles = \
#         run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)
#
#     print(algs_acc_costs)
#
#     plot_history(algs_acc_costs)
#
#     plt.savefig(f"history_plots/history_{name}_orig_without_mindet.pdf")
#     plt.show()


# def run_and_plot_history_mindet_sections(phi, prices, demands, alg_list, with_mindet, gamma, name):
#     algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles = \
#         run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)
#
#     print(algs_acc_costs)
#
#     plot_history_mindet_sections(algs_acc_costs, mindet_purchases, mindet_current_algs, mindet_cycles, phi, gamma, name)
#
#     plt.savefig(f"history_plots/history_{name}_mindet_sections_without_hlines.pdf")
#     plt.show()


def get_breakpoints(index_list):
    """
    Traverses a given list (index_list) of indices and generates a new list (breakpoints) of the same length.
    If an element in index_list differs from its preceding element, it sets the corresponding element in
    breakpoints to 1, otherwise 0.
    :param index_list: List of numbers
    :return: list of 0 and 1
    """
    breakpoints = [0] * len(index_list)
    for i in range(1, len(index_list)):
        if index_list[i] != index_list[i-1]:
            breakpoints[i] = 1

    return breakpoints


def get_lengths_of_chunks(index_list):
    """
    Returns a list of integers.
    Every integer corresponds to the number of how many times the same number appears in index_list.
    :param index_list: List of numbers
    :return: List of integers
    """
    if len(index_list) == 0:
        return [0]
    lengths_of_chunks = []
    counter = 1
    for i in range(1, len(index_list)):
        if index_list[i] != index_list[i - 1]:
            lengths_of_chunks.append(counter)
            counter = 0
        counter += 1

    lengths_of_chunks.append(counter)

    return lengths_of_chunks


def get_ends_of_chunks(index_list):
    """
    Returns a list of integers.
    Every integer corresponds to the number of how many times the same number appears in index_list.
    :param index_list: List of numbers
    :return: List of integers
    """
    if len(index_list) == 0:
        return [0]
    ends_of_chunks = []
    for i in range(1, len(index_list)):
        if index_list[i] != index_list[i - 1]:
            ends_of_chunks.append(i-1)

    ends_of_chunks.append(len(index_list)-1)

    return ends_of_chunks


class History:
    def __init__(self, gamma, phi, prices, demands, alg):
        self.gamma = gamma
        self.phi = phi
        self.prices = prices
        self.demands = demands
        self.alg = alg
        self.purchases = []
        self.acc_costs = []
        self.stocks = []
        # opt_pred = pred.opt_off(prices, demands)
        # self.opt_off = FtP(0, 0, opt_pred)

    def update_purchases(self):
        self.purchases.append(self.alg.x)

    def update_acc_costs(self):
        self.acc_costs.append(self.alg.cost)

    def update_stocks(self):
        self.stocks.append(self.alg.stock)

    # def update_stock_errors(self):
    #     self.stock_errors.append(np.abs(self.opt_off.stock - self.alg.stock))

    def run_step(self, gamma, phi, price, demand):
        self.alg.run(gamma, phi, price, demand)
        self.update_purchases()
        self.update_acc_costs()
        self.update_stocks()

    def run_full(self):
        for i in range(len(self.prices)):
            self.run_step(self.gamma, self.phi, self.prices[i], self.demands[i])
            # self.opt_off.run(self.gamma, self.phi, self.prices[i], self.demands[i])
            # self.update_stock_errors(self.opt_off)

    def get_cost(self):
        return self.alg.cost

    def get_comp_ratio(self, opt_off):
        return self.alg.cost / opt_off.alg.cost

    def get_additive_term(self, opt_off):
        """
        The algorithm incurs a cost of the form "cost(OFF) + x"
        :return: the value of x
        """
        return self.alg.cost - opt_off.alg.cost

    def get_stock_error(self, opt_off):
        if isinstance(self.alg, AlgorithmPred):
            return np.sum(np.abs(self.alg.predictions[i] - opt_off.stocks[i]) for i in range(len(opt_off.stocks)))
        else:
            print("Algorithm does not use predictions.")
            return 0

    def get_normalized_stock_error(self, opt_off):
        return self.get_stock_error(opt_off) / opt_off.cost

    def get_purchase_error(self, opt_off):
        if isinstance(self.alg, AlgorithmPred):
            return np.sum(np.abs(self.purchases[i] - opt_off.purchases[i]) for i in range(len(opt_off.purchases)))
        else:
            print("Algorithm is not an algorithm using predictions.")
            return 0

    def get_normalized_purchase_error(self, opt_off):
        return self.get_purchase_error(opt_off) / opt_off.cost

    def generate_history_df(self):

        len_input = len(self.prices)  # length of the input sequence
        columns = list(range(len_input))

        df = pd.concat([pd.DataFrame(columns=columns),
                        pd.DataFrame([self.prices], columns=columns, index=["Price"]),
                        pd.DataFrame([self.demands], columns=columns, index=["Demand"])])

        df = df.append(pd.DataFrame([map(str, list(self.purchases))], columns=columns, index=[f"$ x(A) $"]))
        df = df.append(pd.DataFrame([list(self.acc_costs)], columns=columns, index=[f"$ \cost(A) $"]))
        df = df.append(pd.DataFrame([list(self.stocks)], columns=columns, index=[f"$ \stock(A) $"]))

        df = df.applymap(lambda x: remove_trailing_zeros(x))

        return df

    def print_history_latex(self):
        df = self.generate_history_df()
        print(df.to_latex(index=True, escape=False))

    def print_history_table(self):

        len_input = len(self.prices)  # length of the input sequence

        output_table = PrettyTable([""] + list(range(len_input)))

        output_table.add_row(["Price"] + list(self.prices))
        output_table.add_row(["Demand"] + list(self.demands))

        output_table.add_row([f"x(ALG)"] + list(self.purchases))
        output_table.add_row([f"cost(ALG)"] + list(self.acc_costs))
        output_table.add_row([f"stock(ALG)"] + list(self.stocks))

        print(output_table)

    def plot_history_costs(self, name="$A$", color="k"):
        len_input = len(self.acc_costs)

        plt.plot(self.acc_costs, '-o', color=color)
        plt.annotate(name, (len_input - .2, self.acc_costs[-1]), color=color)

        plt.xlabel("Time step $ t $")
        plt.ylabel("Accumulated Costs $ cost_t $")

        plt.xlim([0, len_input + 1])


# class HistoryPred(History):
#     def __init__(self, gamma, phi, prices, demands, alg, predictions):
#         super().__init__(gamma, phi, prices, demands, alg)
#         self.predictions = predictions
#         self.stock_errors = []
#
#     def update_stock_errors(self, prediction):
#         self.stock_errors.append(np.abs(self.opt_off.stock - prediction))
#
#     def run_full(self):
#         for i in range(len(self.prices)):
#             self.run_step(self.gamma, self.phi, self.prices[i], self.demands[i])
#             self.opt_off.run(self.gamma, self.phi, self.prices[i], self.demands[i])
#             self.update_stock_errors(self.predictions[i])
#
#     def get_total_error(self):
#         return np.sum(self.stock_errors)
#
#     def get_normalized_error(self):
#         return self.get_total_error() / self.opt_off.cost
#
#     def get_results_str(self):
#         return f"cost = {self.get_cost()}, off cost = {self.opt_off.cost}, competitive ratio = {self.get_comp_ratio()},\
#                  eta = {self.get_total_error()}"


class MinDetHistory:
    def __init__(self, gamma, phi, prices, demands, alg_list):
        self.gamma = gamma
        self.phi = phi
        self.prices = prices
        self.demands = demands
        self.algs_histories = [History(gamma, phi, prices, demands, alg) for alg in alg_list]
        self.mindet_history = History(gamma, phi, prices, demands, MinDet(0, 0, [a.alg for a in self.algs_histories]))
        self.mindet_current_algs = []
        self.mindet_cycles = []

    def run_step(self, gamma, phi, price, demand):
        self.mindet_history.alg.run(gamma, phi, price, demand)
        self.mindet_current_algs.append(self.mindet_history.alg.current_alg)

        for alg in self.algs_histories:

            alg.update_purchases()
            alg.update_acc_costs()
            alg.update_stocks()

        self.mindet_history.update_purchases()
        self.mindet_history.update_acc_costs()
        self.mindet_history.update_stocks()
        self.mindet_cycles.append(self.mindet_history.alg.cycle)

    def run_full(self):
        for i in range(len(self.prices)):
            self.run_step(self.gamma, self.phi, self.prices[i], self.demands[i])
            # self.opt_off.run(self.gamma, self.phi, self.prices[i], self.demands[i]) TODO

    def generate_history_df(self):

        len_input = len(self.prices)  # length of the input sequence
        columns = list(range(len_input))

        df = pd.concat([pd.DataFrame(columns=columns),
                        pd.DataFrame([self.prices], columns=columns, index=["Price"]),
                        pd.DataFrame([self.demands], columns=columns, index=["Demand"])])

        for i, alg in enumerate(self.algs_histories):
            df = df.append(pd.DataFrame([map(str, list(alg.purchases))], columns=columns, index=[f"$ x(A_{i}) $"]))
            df = df.append(pd.DataFrame([list(alg.acc_costs)], columns=columns, index=[f"$ \cost(A_{i}) $"]))
            df = df.append(pd.DataFrame([list(alg.stocks)], columns=columns, index=[f"$ \stock(A_{i}) $"]))

        df = df.append(pd.DataFrame([self.mindet_history.purchases], columns=columns, index=["$ x(\mindet) $"]))
        df = df.append(pd.DataFrame([self.mindet_history.acc_costs], columns=columns, index=["$ \cost(\mindet) $"]))
        df = df.append(pd.DataFrame([self.mindet_history.stocks], columns=columns, index=["$ \stock(\mindet) $"]))
        df = df.append(
            pd.DataFrame([self.mindet_current_algs], columns=columns, index=["Alg. currently exec. by $ \mindet $"]))

        df = df.applymap(lambda x: remove_trailing_zeros(x))

        return df

    def print_history_table(self):

        len_input = len(self.prices)  # length of the input sequence

        output_table = PrettyTable([""] + list(range(len_input)))

        output_table.add_row(["Price"] + list(self.prices))
        output_table.add_row(["Demand"] + list(self.demands))

        for i, alg in enumerate(self.algs_histories):
            output_table.add_row([f"x(ALG_{i})"] + list(alg.purchases))
            output_table.add_row([f"cost(ALG_{i})"] + list(alg.acc_costs))
            output_table.add_row([f"stock(ALG_{i})"] + list(alg.stocks))

        output_table.add_row(["x(MIN^det)"] + list(self.mindet_history.purchases))
        output_table.add_row(["cost(MIN^det)"] + list(self.mindet_history.acc_costs))
        output_table.add_row(["Current_Alg"] + list(self.mindet_current_algs))
        output_table.add_row(["stock(MIN^det)"] + list(self.mindet_history.stocks))
        # output_table.add_row(["Ratio cost(MIN^det)/min(cost(A_i))"] + [
        #     self.mindet_history.acc_costs /
        # TODO

        print(output_table)

    def plot_history_costs(self, with_color=True):
        len_input = len(self.algs_histories[0].acc_costs)

        if with_color:
            for i, alg in enumerate(self.algs_histories):
                plt.plot(alg.acc_costs, '-o', color=color_list[i])
                plt.annotate(f"$A_{i}$", (len_input - .2, alg.acc_costs[-1]), color=color_list[i])
        else:
            for i, alg in enumerate(self.algs_histories):
                plt.plot(alg.acc_costs, '-o', color="k")
                plt.annotate(f"$A_{i}$", (len_input - .2, alg.acc_costs[-1]), color="k")

        plt.xlabel("Time step $ t $")
        plt.ylabel("Accumulated Costs $ cost_t $")

        plt.xlim([0, len_input + 1])

    def plot_history_mindet_sections(self):
        num_algs = len(self.algs_histories)

        self.plot_history_costs(with_color=False)

        ends_chunks = get_ends_of_chunks(self.mindet_cycles)

        start = 0
        for l in range(len(ends_chunks)):
            end = ends_chunks[l]
            current_alg = self.mindet_current_algs[end]
            plt.plot(range(start, end + 1), self.algs_histories[current_alg].acc_costs[start:end + 1], '-o',
                     color=color_list[current_alg], label=f"$ A_{current_alg} $")
            start = end

        # dotted horizontal lines
        for l in range(max(self.mindet_cycles)):
            plt.hlines((l + 1) * self.gamma * self.phi, 0, len(self.mindet_current_algs), linestyle='dotted', color=color_list[l % num_algs], linewidth=0.5)

        # solid horizontal lines
        for l in self.mindet_cycles:
            plt.hlines((l + 1) * self.gamma * self.phi, 0, len(self.mindet_current_algs), color=color_list[l % num_algs], linewidth=0.8)

        plt.xticks(ends_chunks)
        plt.yticks([(l + 1) * self.gamma * self.phi for l in self.mindet_cycles])

        # history
        lines = []
        for current_alg in range(num_algs):
            line, = plt.plot(range(1), range(1), '-o', color=color_list[current_alg],
                             label="MIN$^{det}$ follows" + f"$ A_{current_alg} $")
            lines.append(line, )
        labels = [f"follow $ A_{current_alg} $" for current_alg in range(num_algs)]
        plt.legend(lines, labels)


class MinRandHistory:
    def __init__(self, gamma, phi, prices, demands, alg_list, eps):
        self.gamma = gamma
        self.phi = phi
        self.prices = prices
        self.demands = demands
        self.eps = eps
        self.algs_histories = [History(gamma, phi, prices, demands, alg) for alg in alg_list]
        self.minrand_history = History(gamma, phi, prices, demands,
                                       MinRand(0, 0, [a.alg for a in self.algs_histories], eps))
        self.minrand_current_algs = []

    def run_step(self, gamma, phi, price, demand):
        self.minrand_history.alg.run(gamma, phi, price, demand)
        self.minrand_current_algs.append(self.minrand_history.alg.current_alg)

        for alg in self.algs_histories:

            alg.update_purchases()
            alg.update_acc_costs()
            alg.update_stocks()

        self.minrand_history.update_purchases()
        self.minrand_history.update_acc_costs()
        self.minrand_history.update_stocks()

    def run_full(self):
        for i in range(len(self.prices)):
            self.run_step(self.gamma, self.phi, self.prices[i], self.demands[i])
            # self.opt_off.run(self.gamma, self.phi, self.prices[i], self.demands[i]) TODO

    def generate_history_df(self):

        len_input = len(self.prices)  # length of the input sequence
        columns = list(range(len_input))

        df = pd.concat([pd.DataFrame(columns=columns),
                        pd.DataFrame([self.prices], columns=columns, index=["Price"]),
                        pd.DataFrame([self.demands], columns=columns, index=["Demand"])])

        for i, alg in enumerate(self.algs_histories):
            df = df.append(pd.DataFrame([map(str, list(alg.purchases))], columns=columns, index=[f"$ x(A_{i}) $"]))
            df = df.append(pd.DataFrame([list(alg.acc_costs)], columns=columns, index=[f"$ \cost(A_{i}) $"]))
            df = df.append(pd.DataFrame([list(alg.stocks)], columns=columns, index=[f"$ \stock(A_{i}) $"]))

        df = df.append(pd.DataFrame([self.minrand_history.purchases], columns=columns, index=["$ x(\mindet) $"]))
        df = df.append(pd.DataFrame([self.minrand_history.acc_costs], columns=columns, index=["$ \cost(\mindet) $"]))
        df = df.append(pd.DataFrame([self.minrand_history.stocks], columns=columns, index=["$ \stock(\mindet) $"]))
        df = df.append(
            pd.DataFrame([self.minrand_current_algs], columns=columns, index=["Alg. currently exec. by $ \mindet $"]))

        df = df.applymap(lambda x: remove_trailing_zeros(x))

        return df

    def print_history_table(self):

        len_input = len(self.prices)  # length of the input sequence

        output_table = PrettyTable([""] + list(range(len_input)))

        output_table.add_row(["Price"] + list(self.prices))
        output_table.add_row(["Demand"] + list(self.demands))

        for i, alg in enumerate(self.algs_histories):
            output_table.add_row([f"x(ALG_{i})"] + list(alg.purchases))
            output_table.add_row([f"cost(ALG_{i})"] + list(alg.acc_costs))
            output_table.add_row([f"stock(ALG_{i})"] + list(alg.stocks))

        output_table.add_row(["x(MIN^det)"] + list(self.minrand_history.purchases))
        output_table.add_row(["cost(MIN^det)"] + list(self.minrand_history.acc_costs))
        output_table.add_row(["Current_Alg"] + list(self.minrand_current_algs))
        output_table.add_row(["stock(MIN^det)"] + list(self.minrand_history.stocks))
        # output_table.add_row(["Ratio cost(MIN^det)/min(cost(A_i))"] + [
        #     self.mindet_history.acc_costs /
        # TODO

        print(output_table)

    def plot_history_costs(self, with_color=True):
        len_input = len(self.algs_histories[0].acc_costs)

        if with_color:
            for i, alg in enumerate(self.algs_histories):
                plt.plot(alg.acc_costs, '-o', color=color_list[i])
                plt.annotate(f"$A_{i}$", (len_input - .2, alg.acc_costs[-1]), color=color_list[i])
        else:
            for i, alg in enumerate(self.algs_histories):
                plt.plot(alg.acc_costs, '-o', color="k")
                plt.annotate(f"$A_{i}$", (len_input - .2, alg.acc_costs[-1]), color="k")

        plt.xlabel("Time step $ t $")
        plt.ylabel("Accumulated Costs $ cost_t $")

        plt.xlim([0, len_input + 1])


# class Comparison:
#     def __init__(self, alg0, alg1):
#         self.alg0 = alg0
#         self.alg1 = alg1
#
#     def compute_error
