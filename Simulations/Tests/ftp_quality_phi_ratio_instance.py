import Simulations.predictions as pred
from Simulations.algorithms import *
import Simulations.offline as off
from Simulations.history import *
import Simulations.instances as inst
import base

import numpy as np
from numpy.random import default_rng
import Simulations.pickle_helpers as ph


# def quality_of_FtP_phi_ratio(prices, demands, num_repetitions, prediction_type="normal deviation"):
#     # outline:
#     # - generate all instances as before
#     # - subdivide errors in evenly sized subranges
#     # - for every phi and every error subrange, compute average competitive ratio
#
#     pred_opt_off = pred.opt_off(prices, demands)
#     phi = np.max(prices)
#     input_length = len(prices)
#     phi_list = np.linspace(1, 1000, 100)
#
#     eta1 = []
#     eta2 = []
#     ratios = []
#     phis = []
#
#     for phi in phi_list:
#         for i in range(num_repetitions):
#             alg_list, opt = base.create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands)
#
#             for alg in alg_list:
#                 alg.run_full()
#
#                 phis.append(phi)
#
#                 e1 = alg.get_stock_error(opt)
#                 eta1.append(e1)
#
#                 e2 = alg.get_purchase_error(opt)
#                 eta2.append(e2)
#
#                 ratios.append(alg.get_comp_ratio(opt))
#
#     return phis, eta1, eta2, ratios
#
#
# def quality_of_FtP_just_additive_terms(prices, demands, num_repetitions, prediction_type="normal deviation"):
#     pred_opt_off = pred.opt_off(prices, demands)
#     phi = np.max(prices)
#     input_length = len(prices)
#
#     eta1 = []
#     eta2 = []
#     additive_terms = []
#
#     for i in range(num_repetitions):
#         alg_list, opt = base.create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands)
#
#         for alg in alg_list:
#             alg.run_full()
#
#             e1 = alg.get_stock_error(opt)
#             eta1.append(e1)
#
#             e2 = alg.get_purchase_error(opt)
#             eta2.append(e2)
#
#             additive_terms.append(alg.get_additive_term(opt))
#
#     return eta1, eta2, additive_terms


if __name__ == '__main__':
    input_len = 100
    num_repetitions = 10  # how often we run one iteration with fixed inputs
    num_generations = 100  # how often we generate new inputs
    phi_list = np.linspace(1, 100, 10)

    treatments_input = ['normal', 'uniform']
    treatments_predictions = ['normal', 'uniform', '0', '1']

    treatment = 'at'  # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

    if treatment == 'cr':
        base.evaluate_phi_ratio(base.quality_of_ftp, treatments_input, treatments_predictions, num_generations,
                                num_repetitions, input_len, phi_list, "FtP-quality-phi-ratio")

    else:
        base.evaluate_phi_ratio(base.quality_of_ftp_just_additive_terms, treatments_input, treatments_predictions,
                                num_generations, num_repetitions, input_len, phi_list,
                                "FtP-quality-phi-ratio-additive-terms")
