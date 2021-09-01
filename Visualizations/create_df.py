import pandas as pd
import numpy as np
import os


def create_df_from_csv(input_file):
    df = pd.read_csv('csv/' + input_file + '.txt', header=None)
    return df


def create_df_worst(phi, m, n_cycles):
    accumulated_costs = np.zeros(m)
    cost_data = np.zeros([m, n_cycles])
    for i in range(m):
        for j in range(n_cycles):
            if i == j % m:
                round_cost = (j + 1) - accumulated_costs[i]
            else:
                round_cost = 1 / phi
            accumulated_costs[i] += round_cost
            cost_data[i, j] = round_cost
    df = pd.DataFrame(cost_data)
    return df


def create_df_worst_with_min_det(phi, m, n_cycles):
    accumulated_costs = np.zeros(m)
    cost_data = np.zeros([m+1, n_cycles])
    for i in range(m):
        for j in range(n_cycles):
            if i == j % m:
                round_cost = (j + 1) - accumulated_costs[i]
                cost_data[m, j] =round_cost
            else:
                round_cost = 1 / phi
            accumulated_costs[i] += round_cost
            cost_data[i, j] = round_cost
    df = pd.DataFrame(cost_data)
    return df


def create_df_equal(phi, m, n_cycles):
    accumulated_costs = np.zeros(m)
    cost_data = np.zeros([m, n_cycles])
    for i in range(m):
        for j in range(n_cycles):
            round_cost = 1
            accumulated_costs[i] += round_cost
            cost_data[i, j] = round_cost
    df = pd.DataFrame(cost_data)
    return df


def create_df_equal_with_min_det(phi, m, n_cycles):
    return create_df_equal(phi, m+1, n_cycles)


def upper_bound_worst_case(phi, m, n_cycles):
    accumulated_costs = np.zeros(m)
    cost_data = np.zeros([m, n_cycles])
    for i in range(m):
        for j in range(n_cycles):
            if i == j % m:
                round_cost = (j + 1) - accumulated_costs[i]
            else:
                round_cost = 0
            accumulated_costs[i] += round_cost
            cost_data[i, j] = round_cost
    df = pd.DataFrame(cost_data)
    return df


def upper_bound_worst_case_with_min_det(phi, m, n_cycles):
    accumulated_costs = np.zeros(m)
    cost_data = np.zeros([m + 1, n_cycles])
    for i in range(m):
        for j in range(n_cycles):
            if i == j % m:
                round_cost = (j + 1) - accumulated_costs[i]
                cost_data[m, j] = round_cost
            else:
                round_cost = 0
            accumulated_costs[i] += round_cost
            cost_data[i, j] = round_cost
    df = pd.DataFrame(cost_data)
    return df

