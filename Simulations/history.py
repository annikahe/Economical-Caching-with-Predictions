from Simulations.algorithms import *
from Simulations.instances import *
import Simulations.offline as off
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd

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

def run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma=1):
    """
        Executes given algorithms on the complete input sequence.
        Can run the algorithms independently from each other or as input for MIN^det.

        :param gamma:
        :param phi:
        :param prices:
        :param demands:
        :param alg_list: list
            List of algorithms for the Economical Caching Problem.
        :param with_mindet: boolean
            Decides whether to simulate the algorithms with or without MIN^det.
            with_mindet = True: Generate an instance of MIN^det with the given algorithms as input.
                                Run this algorithm on the given input sequence and generate the history.
            with_mindet = False: Only run the given algorithms (independently from each other)
                                and generate the history.
        :return:
            - numpy array of arrays: inner arrays contain the purchase amounts of all input algorithms
                                     in the individual time steps.
            - numpy array of arrays: inner arrays contain the costs the different input algorithms
                                     accumulated until the individual time steps.
            - numpy array of arrays: inner arrays contain the stock levels of the different input algorithms
                                     in the individual time steps.
        """
    algs_purchases = [[] for i in range(len(alg_list))]
    algs_acc_costs = [[] for i in range(len(alg_list))]
    algs_stocks = [[] for i in range(len(alg_list))]

    if with_mindet:
        mindet = MinDet(0, 0, alg_list)

    mindet_purchases = []
    mindet_current_algs = []
    mindet_acc_costs = []
    mindet_stocks = []
    mindet_cycles = []

    for i in range(len(prices)):
        price = prices[i]
        demand = demands[i]
        if with_mindet:
            mindet.run(gamma, phi, price, demand)
            mindet_current_algs.append(mindet.current_alg)
        for j, alg in enumerate(alg_list):
            if not with_mindet:
                alg.run(gamma, phi, price, demand)
            algs_purchases[j].append(alg.x)
            algs_acc_costs[j].append(alg.cost)
            algs_stocks[j].append(alg.stock)
        if with_mindet:
            mindet_purchases.append(mindet.x)
            mindet_acc_costs.append(mindet.cost)
            mindet_stocks.append(mindet.stock)
            mindet_cycles.append(mindet.cycle)

    return np.array(algs_purchases), np.array(algs_acc_costs), np.array(algs_stocks), mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles


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


def run_and_generate_history_df(phi, prices, demands, alg_list, with_mindet, gamma=1):
    """
    Run the algorithms given in alg_list
    :param gamma:
    :param phi:
    :param prices:
    :param demands:
    :param alg_list:
    :param with_mindet:
    :return:
    """
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks = \
        run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)

    len_input = len(prices)  # length of the input sequence
    columns = list(range(len_input))

    df = pd.concat([pd.DataFrame(columns=columns),
                    pd.DataFrame([prices], columns=columns, index=["Price"]),
                    pd.DataFrame([demands], columns=columns, index=["Demand"])])

    for i in range(len(algs_purchases)):
        df = df.append(pd.DataFrame([map(str, list(algs_purchases[i]))], columns=columns, index=[f"$ x(A_{i}) $"]))
        df = df.append(pd.DataFrame([list(algs_acc_costs[i])], columns=columns, index=[f"$ \cost(A_{i}) $"]))
        df = df.append(pd.DataFrame([list(algs_stocks[i])], columns=columns, index=[f"$ \stock(A_{i}) $"]))

    if with_mindet:
        df = df.append(pd.DataFrame([mindet_purchases], columns=columns, index=["$ x(\mindet) $"]))
        df = df.append(pd.DataFrame([mindet_acc_costs], columns=columns, index=["$ \cost(\mindet) $"]))
        df = df.append(pd.DataFrame([mindet_stocks], columns=columns, index=["$ \stock(\mindet) $"]))
        df = df.append(pd.DataFrame([mindet_current_algs], columns=columns, index=["Alg. currently exec. by $ \mindet $"]))

    df = df.applymap(lambda x: remove_trailing_zeros(x))

    if with_mindet:
        for i in range(len_input):
            if int(df.at["Alg. currently exec. by $ \mindet $", i]) == 0:
                df.at["$ x(A_0) $", i] = color_cell_latex(df.at["$ x(A_0) $", i])
            elif int(df.at["Alg. currently exec. by $ \mindet $", i]) == 1:
                df.at["$ x(A_1) $", i] = color_cell_latex(df.at["$ x(A_1) $", i])

    return df


def run_and_print_history_latex(phi, prices, demands, alg_list, with_mindet, gamma=1):
    df = run_and_generate_history_df(phi, prices, demands, alg_list, with_mindet, gamma)
    print(df.to_latex(index=True, escape=False))


def run_and_print_history_table(phi, prices, demands, alg_list, with_mindet, gamma=1):
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks \
        = run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)

    len_input = len(prices)  # length of the input sequence

    output_table = PrettyTable([""] + list(range(len_input)))

    output_table.add_row(["Price"] + list(prices))
    output_table.add_row(["Demand"] + list(demands))

    for i in range(len(alg_list)):
        output_table.add_row([f"x(ALG_{i})"] + list(algs_purchases[i]))
        output_table.add_row([f"cost(ALG_{i})"] + list(algs_acc_costs[i]))
        output_table.add_row([f"stock(ALG_{i})"] + list(algs_stocks[i]))

    if with_mindet:
        output_table.add_row(["x(MIN^det)"] + list(mindet_purchases))
        output_table.add_row(["cost(MIN^det)"] + list(mindet_acc_costs))
        output_table.add_row(["Current_Alg"] + list(mindet_current_algs))
        output_table.add_row(["stock(MIN^det)"] + list(mindet_stocks))
        output_table.add_row(["Ratio cost(MIN^det)/min(cost(A_i))"] + [mindet_acc_costs[i]/min(algs_acc_costs[0][i], algs_acc_costs[1][i]) for i in range(len(prices))])

    print(output_table)


def plot_history(algs_quantity, with_color=True):
    """
    Plots the development of a certain quantity of the given algorithms.
    :param algs_quantity: List
        A list with the values of some quantity of the online algorithms (e.g. the accumulated cost in every step)..
    :param with_color: Boolean
        Determines whether to plot the quantities in different colors or not.
    """
    num_algs = len(algs_quantity)
    len_input = len(algs_quantity[0])

    if with_color:
        for i in range(num_algs):
            plt.plot(algs_quantity[i], '-o', color=color_list[i])
            plt.annotate(f"$A_{i}$", (len_input - .2, algs_quantity[i][-1]), color=color_list[i])
    else:
        for i in range(num_algs):
            plt.plot(algs_quantity[i], '-o', color="k")
            plt.annotate(f"$A_{i}$", (len_input - .2, algs_quantity[i][-1]), color=color_list[i])

    plt.xlabel("Time step $ t $")
    plt.ylabel("Accumulated Costs $ cost_t $")

    plt.xlim([0, len_input + 1])


def plot_history_mindet_sections(algs_quantity, mindet_purchases, mindet_current_algs, mindet_cycles, phi, gamma, name):
    num_algs = len(algs_quantity)

    plot_history(algs_quantity, with_color=False)

    ends_chunks = get_ends_of_chunks(mindet_cycles)

    start = 0
    for l in range(len(ends_chunks)):
        end = ends_chunks[l]
        current_alg = mindet_current_algs[end]
        plt.plot(range(start, end+1), algs_quantity[current_alg][start:end + 1], '-o', color=color_list[current_alg], label=f"$ A_{current_alg} $")
        start = end

    # dotted horizontal lines
    # for l in range(max(mindet_cycles)):
    #     plt.hlines((l + 1) * gamma * phi, 0, len(mindet_current_algs), linestyle='dotted', color=color_list[l % num_algs], linewidth=0.5)

    # solid horizontal lines
    # for l in mindet_cycles:
    #     plt.hlines((l + 1) * gamma * phi, 0, len(mindet_current_algs), color=color_list[l % num_algs], linewidth=0.8)

    plt.xticks(ends_chunks)
    plt.yticks([(l + 1) * gamma * phi for l in mindet_cycles])

    # history
    lines = []
    for current_alg in range(num_algs):
        line, = plt.plot(range(1), range(1), '-o', color=color_list[current_alg], label="MIN$^{det}$ follows" + f"$ A_{current_alg} $")
        lines.append(line,)
    labels = [f"follow $ A_{current_alg} $" for current_alg in range(num_algs)]
    plt.legend(lines, labels)


def run_and_plot_history(phi, prices, demands, alg_list, with_mindet, gamma, name):
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles = \
        run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)

    print(algs_acc_costs)

    plot_history(algs_acc_costs)

    plt.savefig(f"history_plots/history_{name}_orig_without_mindet.png")
    plt.show()


def run_and_plot_history_mindet_sections(phi, prices, demands, alg_list, with_mindet, gamma, name):
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles = \
        run_and_generate_history(phi, prices, demands, alg_list, with_mindet, gamma)

    print(algs_acc_costs)

    plot_history_mindet_sections(algs_acc_costs, mindet_purchases, mindet_current_algs, mindet_cycles, phi, gamma, name)

    plt.savefig(f"history_plots/history_{name}_mindet_sections_without_hlines.png")
    plt.show()


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
