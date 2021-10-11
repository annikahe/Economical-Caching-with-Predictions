from operator import itemgetter
from decimal import *
import numpy as np


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
                        pp[2] -= min(purchase_amount, pp[2])  # decrease remaining stock levels of earlier and later steps up to current time step
                    if p[0] < pp[0] < current_step: # increase stock levels of later time steps up to current time step
                        pp[3] += min(purchase_amount, 1 - pp[3])
                    elif pp[0] == current_step:  # increase level of current stock
                        pp[2] += min(purchase_amount, 1 - pp[2])
        else:
            break

    purchases.sort(key=itemgetter(0))


def opt_decisions(prices, demands):
    # data structure: list of tuples
    # (time step, price, remaining space in stock, stock level)

    getcontext().prec = 16

    purchase_overview = [list(a) for a in zip(map(Decimal, range(len(prices))), prices, len(prices) * [1], len(prices) * [0])]

    old_remaining_stock = Decimal(1)
    old_stock = Decimal(0)

    for t in range(len(prices)):
        remaining_demand = demands[t]
        purchase_overview[t][2] = old_remaining_stock
        purchase_overview[t][3] = old_stock
        take_from_stock(purchase_overview, Decimal(remaining_demand), t, Decimal(prices[t]))
        old_remaining_stock = purchase_overview[t][2]
        old_stock = purchase_overview[t][3]

    opt_stock_levels = [float(p[3]) for p in purchase_overview]

    return opt_stock_levels


def compute_error(list1, list2):
    error = 0
    for i in range(len(list1)):
        error += np.abs(list1[i] - list2[i])

    return error


# opt_decisions([2,1,3,4,5,1], [0,0,0.2,1.6,0.4,0.2])

# opt_decisions([5,4,3,2,1,1,2,3,4,5], [0.1, 0.1, 0.2, 0, 0, 0,0.2, 0.1, 0.5, 0.5])



