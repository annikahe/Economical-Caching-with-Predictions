import Simulations.predictions as pred
from Simulations.algorithms import *
import Simulations.offline as off
from Simulations.history import *
import Simulations.instances as inst
import base

import numpy as np
from numpy.random import default_rng
import Simulations.pickle_helpers as ph


if __name__ == '__main__':
    input_len = 100
    phi = 100
    num_repetitions = 10  # how often we run one iteration with fixed inputs
    num_generations = 100  # how often we generate new inputs

    treatments_input = ['normal', 'uniform']
    treatments_predictions = ['normal', 'uniform', '0', '1']

    treatment = 'at' # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

    if treatment == 'cr':
        base.evaluate_error_ratio(base.quality_of_ftp, treatments_input, treatments_predictions, num_generations,
                                  num_repetitions, input_len, phi, "FtP-quality")

    else:
        base.evaluate_error_ratio(base.quality_of_ftp_just_additive_terms, treatments_input, treatments_predictions,
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
