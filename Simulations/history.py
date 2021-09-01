from Simulations.algorithms import *
from Simulations.instances import *
import Simulations.offline as off
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd


def run_and_generate_history(gamma, phi, prices, consumptions, alg_list, with_mindet):
    algs_purchases = [[] for i in range(len(alg_list))]
    algs_acc_costs = [[] for i in range(len(alg_list))]
    algs_stocks = [[] for i in range(len(alg_list))]

    if with_mindet:
        mindet = MinDet(0, 0, alg_list)

    mindet_purchases = []
    mindet_current_algs = []
    mindet_acc_costs = []
    mindet_stocks = []

    for i in range(len(prices)):
        price = prices[i]
        consumption = consumptions[i]
        if with_mindet:
            mindet.run(gamma, phi, price, consumption)
            mindet_current_algs.append(mindet.current_alg)
        for j, alg in enumerate(alg_list):
            if not with_mindet:
                alg.run(gamma, phi, price, consumption)
            algs_purchases[j].append(alg.x)
            algs_acc_costs[j].append(alg.cost)
            algs_stocks[j].append(alg.stock)
        if with_mindet:
            mindet_purchases.append(mindet.x)
            mindet_acc_costs.append(mindet.cost)
            mindet_stocks.append(mindet.stock)

    return np.array(algs_purchases), np.array(algs_acc_costs), np.array(algs_stocks), mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks


def remove_trailing_zeros(val):
    val = str(val)
    return val.rstrip('0').rstrip('.') if '.' in val else val


def color_cell_latex(val):
    return "\cellcolor{yellow!50} " + str(val)


def run_and_generate_history_df(gamma, phi, prices, consumptions, alg_list, with_mindet):
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks = run_and_generate_history(
        gamma, phi, prices, consumptions, alg_list, with_mindet)

    len_input = len(prices)  # length of the input sequence
    columns = list(range(len_input))

    df = pd.concat([pd.DataFrame(columns=columns),
                    pd.DataFrame([prices], columns=columns, index=["Price"]),
                    pd.DataFrame([consumptions], columns=columns, index=["Demand"])])

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

    print(df.to_latex(index=True, escape=False))

    return df


def run_and_print_history(gamma, phi, prices, consumptions, alg_list, with_mindet):
    algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks = run_and_generate_history(gamma, phi, prices, consumptions, alg_list, with_mindet)

    len_input = len(prices)  # length of the input sequence

    output_table = PrettyTable([""] + list(range(len_input)))

    output_table.add_row(["Price"] + list(prices))
    output_table.add_row(["Demand"] + list(consumptions))

    for i in range(len(alg_list)):
        output_table.add_row([f"x(ALG_{i})"] + list(algs_purchases[i]))
        output_table.add_row([f"cost(ALG_{i})"] + list(algs_acc_costs[i]))
        output_table.add_row([f"stock(ALG_{i})"] + list(algs_stocks[i]))

    if with_mindet:
        output_table.add_row([f"x(MIN^det)"] + list(mindet_purchases))
        output_table.add_row([f"cost(MIN^det)"] + list(mindet_acc_costs))
        output_table.add_row(["Current_Alg"] + list(mindet_current_algs))
        output_table.add_row(["stock(MIN^det)"] + list(mindet_stocks))
        output_table.add_row(["Ratio cost(MIN^det)/min(cost(A_i))"] + [mindet_acc_costs[i]/min(algs_acc_costs[0][i], algs_acc_costs[1][i]) for i in range(len(prices))])
        epsilon = 1/100
        output_table.add_row(["Lower bound correct?"] + [mindet_acc_costs[i] > (3 - epsilon) * min(algs_acc_costs[0][i], algs_acc_costs[1][i]) for i in range(len(prices))])

    print(output_table)


def plot_history(algs_purchases, mindet_purchases, mindet_current_algs):
    # TODO
    for i in range(len(algs_purchases)):
        plt.plot(algs_purchases[i], '-o', color="k")
    #plt.plot(mindet_purchases, 'o', color="red")

    ends_chunks = get_ends_of_chunks(mindet_current_algs)
    start = 0
    for l in range(len(ends_chunks)):
        end = ends_chunks[l]
        current_alg = mindet_current_algs[start]
        print(range(start, end))
        plt.plot(range(start, end), algs_purchases[current_alg][start:end], '-o', color="red")
        start = ends_chunks[l] - 1
        print(l)
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

    ends_of_chunks.append(i)

    return ends_of_chunks
