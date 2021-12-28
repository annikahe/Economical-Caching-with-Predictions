import Simulations.predictions as pred
from Simulations.algorithms import *
import Simulations.offline as off
from Simulations.history import *
import Simulations.instances as inst
import base

import numpy as np
from numpy.random import default_rng
import Simulations.pickle_helpers as ph


def quality_of_FtP(prices, demands, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    ratios = []

    for i in range(num_repetitions):
        rng = default_rng(i)

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

            e1 = alg.get_stock_error(opt)
            eta1.append(e1)

            e2 = alg.get_purchase_error(opt)
            eta2.append(e2)

            ratios.append(alg.get_comp_ratio(opt))

    return eta1, eta2, ratios


def quality_of_FtP_just_additive_terms(prices, demands, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    additive_terms = []

    for i in range(num_repetitions):
        rng = default_rng(i)

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

            e1 = alg.get_stock_error(opt)
            eta1.append(e1)

            e2 = alg.get_purchase_error(opt)
            eta2.append(e2)

            additive_terms.append(alg.get_additive_term(opt))

    return eta1, eta2, additive_terms


if __name__ == '__main__':
    input_len = 100
    phi = 100
    num_repetitions = 10  # how often we run one iteration with fixed inputs
    num_generations = 100  # how often we generate new inputs

    treatments_input = ['normal', 'uniform']
    treatments_predictions = ['normal', 'uniform', '0', '1']

    treatment = 'at' # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

    if treatment == 'cr':
        base.evaluate_error_ratio(quality_of_FtP, treatments_input, treatments_predictions, num_generations,
                                  num_repetitions, input_len, phi, "FtP-quality")

    else:
        base.evaluate_error_ratio(quality_of_FtP_just_additive_terms, treatments_input, treatments_predictions,
                                  num_generations, num_repetitions, input_len, phi, "FtP-quality-additive-terms")

    # # Uniformly generated inputs, predictions deviate normally from optimal offline predictions
    # eta1 = []
    # eta2 = []
    # ratios = []
    # for i in range(num_generations):
    #     prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
    #     demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    #
    #     e1, e2, r = quality_of_FtP(prices, demands, num_repetitions, prediction_type="normal deviation")
    #
    #     eta1.append(e1)
    #     eta2.append(e2)
    #     ratios.append(r)
    #
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-unif_preds-normal.pkl')
    #
    # # Uniformly generated inputs, uniformly generated predictions
    # eta1 = []
    # eta2 = []
    # ratios = []
    # for i in range(num_generations):
    #     prices = inst.prices_uniform(input_len, phi)
    #     demands = inst.demands_uniform(input_len)
    #
    #     e1, e2, r = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="uniform random")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-unif_preds-unif.pkl')
    #
    # # Uniformly generated inputs, predictions all = 0
    #
    # prices = inst.prices_uniform(input_len, phi)
    # demands = inst.demands_uniform(input_len)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="all 0")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-unif_preds-0.pkl')
    #
    # # Uniformly generated inputs, predictions all = 1
    #
    # prices = inst.prices_uniform(input_len, phi)
    # demands = inst.demands_uniform(input_len)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="all 1")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-unif_preds-1.pkl')
    #
    #
    # # Normally generated inputs, predictions deviate normally from optimal offline predictions
    #
    # prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
    # demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions, prediction_type="normal deviation")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-normal_preds-normal.pkl')
    #
    # # Normally generated inputs, uniformly generated predictions
    #
    # prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
    # demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="uniform random")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-normal_preds-unif.pkl')
    #
    # # Normally generated inputs, predictions all = 0
    #
    # prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
    # demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="all 0")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-normal_preds-0.pkl')
    #
    # # Normally generated inputs, predictions all = 1
    #
    # prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
    # demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    #
    # eta1, eta2, ratios = quality_of_FtP(prices, demands, num_repetitions * 10, prediction_type="all 1")
    # ph.save_objects([eta1, eta2, ratios], 'Instances/FtP_quality_input-normal_preds-1.pkl')
