from operator import itemgetter
from decimal import *
import numpy as np
import pandas as pd
from numpy.random import default_rng


# compute optimal offline algorithm

def take_from_stock(purchases, remaining_demand, current_step, current_price):
    purchases.sort(key=itemgetter(1))

    for p in purchases:
        if p[1] < current_price and remaining_demand > 0:
            if p[0] < current_step:
                purchase_amount = Decimal(min(p[2], remaining_demand))
                remaining_demand -= purchase_amount  # decrease remaining demand
                p[3] += purchase_amount  # increase stock level
                p[2] -= purchase_amount  # decrease remaining stock level

                for pp in purchases:
                    if pp[0] < p[0] or p[0] < pp[0] < current_step:
                        pp[2] -= min(purchase_amount, pp[2])
                        # decrease remaining stock levels of earlier and later steps up to current time step
                    if p[0] < pp[0] < current_step:  # increase stock levels of later time steps up to current time step
                        pp[3] += min(purchase_amount, 1 - pp[3])
                    elif pp[0] == current_step:  # increase level of current stock
                        pp[2] += min(purchase_amount, 1 - pp[2])
        else:
            break

    purchases.sort(key=itemgetter(0))


def opt_stock_levels(prices, demands):
    # data structure: list of tuples
    # (time step, price, remaining space in stock, stock level)

    getcontext().prec = 16

    purchase_overview = [list(a) for a in zip(map(Decimal, range(len(prices))), prices,
                                              len(prices) * [1], len(prices) * [0])]

    old_remaining_stock = Decimal(1)
    old_stock = Decimal(0)

    for t in range(len(prices)):
        remaining_demand = demands[t]
        purchase_overview[t][2] = old_remaining_stock
        purchase_overview[t][3] = old_stock
        take_from_stock(purchase_overview, Decimal(remaining_demand), t, Decimal(prices[t]))
        old_remaining_stock = purchase_overview[t][2]
        old_stock = purchase_overview[t][3]

    return [float(p[3]) for p in purchase_overview]


# Error with respect to optimal offline algorithm

def compute_error(opt, onl):
    return np.sum(np.abs(opt[i] - onl[i]) for i in range(len(opt)))


# Upper bound on cost of FtP w.r.t OFF

def bound_FtP_stock_error(phi, off, eta):
    return off + np.min([phi * eta, (phi - 1)*(eta - 1) + phi])


def comp_ratio_FtP_stock_error(phi, off, eta_norm):
    """

    Parameters
    ----------
    phi
    off
    eta_norm: normalized error eta/off

    Returns
    -------
    competitive ratio for FtP for given normalized error (eta/off).
    """
    return 1 + np.min([phi * eta_norm, (phi - 1)*(eta_norm - 1/off) + phi/off])


# plotting results

def moving_average(a, window_size=3):
    ret = np.cumsum(a, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def get_error_and_ratios(errors, ratios, window_size, num_repetitions=1, drop_duplicates=True):
    df = pd.DataFrame({"Errors": errors, "Ratios": ratios})
    df = df.sort_values("Errors")
    if drop_duplicates:
        df = df.drop_duplicates()
    e = df['Errors'].values.tolist()
    r = df['Ratios'].values.tolist()
    for i in range(num_repetitions):
        e = moving_average(e, window_size)
        r = moving_average(r, window_size)
    # print(moving_average(x, 200))
    # print(moving_average(y, 200))

    return e, r


def get_phi_and_ratios(phis, errors, ratios, window_size, num_repetitions=1, drop_duplicates=True):
    df = pd.DataFrame({"Phi": phis, "Errors": errors, "Ratios": ratios})
    # df = df.sort_values("Phi")
    if drop_duplicates:
        df = df.drop_duplicates()
    p = df['Phi'].values.tolist()
    r = df['Ratios'].values.tolist()
    for i in range(num_repetitions):
        p = moving_average(p, window_size)
        r = moving_average(r, window_size)

    return p, r


def create_predictions(method="normal", input_length=0, deviation=0,  rng=default_rng(), pred=[]):
    if method == "normal":  # normal deviation from opt
        return [np.clip(x + rng.normal(0, deviation), 0, 1) for x in pred]
    elif method == "uniform":  # place predictions uniform in [0,1]
        return [rng.uniform() for _ in range(input_length)]
    elif method == "1":
        return input_length * [1]
    elif method == "0":
        return input_length * [0]


# def quality_of_FtP(prices, demands, deviations, num_repetitions):
#     pred_opt_off = opt_stock_levels(prices, demands)
#     phi = np.max(prices)
#     input_length = len(prices)
#
#     eta1 = []
#     eta2 = []
#     ratios = []
#
#     for i in range(num_repetitions):
#         rng = default_rng(i)
#
#         pred_opt_off_copy = pred_opt_off.copy()
#         opt = FtP(0, 0, pred_opt_off_copy)
#
#         alg_list = [opt]
#
#         for d in deviations:
#             #pred_opt_off_distorted = [np.clip(x + rng.normal(0, d), 0, 1) for x in pred_opt_off_copy]
#             pred_opt_off_distorted = create_predictions("normal deviation", input_length, d, rng, pred_opt_off_copy)
#             distorted = FtP(0, 0, pred_opt_off_distorted)
#             alg_list.append(distorted)
#
#         pred_random = create_predictions("uniform random", input_length, rng)
#         distorted = FtP(0, 0, pred_random)
#         alg_list.append(distorted)
#
#         # all 0 seems to be very good
#         pred_0 = create_predictions("all 0", input_length)
#         distorted = FtP(0, 0, pred_0)
#         alg_list.append(distorted)
#
#         # all 1 seems to be not as good as all 0
#         pred_1 = create_predictions("all 1", input_length)
#         distorted = FtP(0, 0, pred_1)
#         alg_list.append(distorted)
#
#         algs_purchases, algs_acc_costs, algs_stocks, mindet_purchases, mindet_current_algs, mindet_acc_costs, mindet_stocks, mindet_cycles \
#             = history.run_and_generate_history(phi, prices, demands, alg_list,
#                                                False)
#         #
#         # print(algs_acc_costs[0][-1])
#
#         for j in range(1, len(alg_list)):
#             e1 = compute_error(algs_stocks[0], algs_stocks[j])
#             e1 /= algs_acc_costs[0][-1]
#             eta1.append(e1)
#
#             e2 = compute_error(algs_purchases[0], algs_purchases[j])
#             e2 /= algs_acc_costs[0][-1]
#             eta2.append(e2)
#
#             ratio = algs_acc_costs[j][-1] / algs_acc_costs[0][-1]
#             ratios.append(ratio)
#
#     return eta1, eta2, ratios


# opt_stock_levels([2,1,3,4,5,1], [0,0,0.2,1.6,0.4,0.2])

# opt_stock_levels([5,4,3,2,1,1,2,3,4,5], [0.1, 0.1, 0.2, 0, 0, 0,0.2, 0.1, 0.5, 0.5])

# TODO: compare optimal offline algorithm with that that always buys 0 or always buys 1
# TODO: compare with completely random algorithm
