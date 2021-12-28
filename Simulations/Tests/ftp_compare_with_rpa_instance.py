import Simulations.predictions as pred
from Simulations.algorithms import *
import Simulations.offline as off
from Simulations.history import *
import Simulations.instances as inst
import base

import numpy as np
from numpy.random import default_rng
import Simulations.pickle_helpers as ph


def quality_of_FtP_phi_ratio(prices, demands, phi, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    # phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    ratios = []

    for i in range(num_repetitions):
        alg_list, opt = base.create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands)

        # rng = default_rng(i)
        #
        # # optimal offline solution
        # pred_opt_off_copy = pred_opt_off.copy()
        # opt = History(1, phi, prices, demands, FtP(0, 0, pred_opt_off_copy))
        # opt.run_full()
        #
        # alg_list = []
        #
        # if prediction_type == "normal":
        #     deviations = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        #     for d in deviations:
        #         # pred_opt_off_distorted = [np.clip(x + rng.normal(0, d), 0, 1) for x in pred_opt_off_copy]
        #         pred_opt_off_distorted = off.create_predictions("normal", input_length, d, rng, pred_opt_off_copy)
        #         alg_list.append(History(1, phi, prices, demands, FtP(0, 0, pred_opt_off_distorted)))
        #
        # else:
        #     predictions = off.create_predictions(prediction_type, input_length, rng)
        #     alg_list.append(History(1, phi, prices, demands, FtP(0, 0, predictions)))

        for alg in alg_list:
            alg.run_full()

            e1 = alg.get_stock_error(opt)
            eta1.append(e1)

            e2 = alg.get_purchase_error(opt)
            eta2.append(e2)

            ratios.append(alg.get_comp_ratio(opt))

    return eta1, eta2, ratios


def quality_of_FtP_and_RPA(prices, demands, phi_list, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    ratios_ftp = []
    ratios_rpa = []

    phis = []

    for phi in phi_list:
        for i in range(num_repetitions):
            rng = default_rng(i)

            # optimal offline solution
            pred_opt_off_copy = pred_opt_off.copy()
            opt = History(1, phi, prices, demands, FtP(0, 0, pred_opt_off_copy))
            opt.run_full()

            alg_list = []

            if prediction_type == "normal":
                deviations = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                for d in deviations:
                    # pred_opt_off_distorted = [np.clip(x + rng.normal(0, d), 0, 1) for x in pred_opt_off_copy]
                    pred_opt_off_distorted = off.create_predictions("normal", input_length, d, rng, pred_opt_off_copy)
                    alg_list.append(History(1, phi, prices, demands, FtP(0, 0, pred_opt_off_distorted)))

            else:
                predictions = off.create_predictions(prediction_type, input_length, rng)
                alg_list.append(History(1, phi, prices, demands, FtP(0, 0, predictions)))

            for alg in alg_list:
                alg.run_full()

                phis.append(phi)

                e1 = alg.get_stock_error(opt)
                eta1.append(e1)

                e2 = alg.get_purchase_error(opt)
                eta2.append(e2)

                ratios_ftp.append(alg.get_comp_ratio(opt))

    return phis, eta1, eta2, ratios_ftp, ratios_rpa



if __name__ == '__main__':
    input_len = 100
    phi_list = np.linspace(1, 1000, 100)
    num_repetitions = 10  # how often we run one iteration with fixed inputs
    num_generations = 100  # how often we generate new inputs

    treatments_input = ['normal', 'uniform']
    treatments_predictions = ['normal', 'uniform', '0', '1']

    treatment = 'at' # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

    phi_list = np.linspace(1, 1000, 100)

    for phi in phi_list:
        if treatment == 'cr':
            base.evaluate_error_ratio(quality_of_FtP_phi_ratio, treatments_input, treatments_predictions, num_generations,
                                      num_repetitions, input_len, phi, "FtP-quality")

        # else:
        #     base.evaluate_error_ratio(quality_of_FtP_just_additive_terms, treatments_input, treatments_predictions,
        #                               num_generations, num_repetitions, input_len, phi, "FtP-quality-additive-terms")