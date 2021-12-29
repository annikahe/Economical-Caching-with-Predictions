import Simulations.instances as inst
import Simulations.pickle_helpers as ph
import Simulations.offline as off
from Simulations.algorithms import *
from Simulations.history import *

import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm


def generate_instance(treatment_instance, input_len, phi):
    if treatment_instance == "normal":
        prices = inst.prices_normal(input_len, phi, mu=1, sigma=0.1)
        demands = inst.demands_normal(input_len, mu=0.5, sigma=0.1)
    elif treatment_instance == "uniform":
        prices = inst.prices_uniform(input_len, phi)
        demands = inst.demands_uniform(input_len)

    return prices, demands


def quality_of_ftp(prices, demands, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    ratios = []

    for i in range(num_repetitions):
        alg_list, opt = create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands)

        for alg in alg_list:
            alg.run_full()

            e1 = alg.get_stock_error(opt)
            eta1.append(e1)

            e2 = alg.get_purchase_error(opt)
            eta2.append(e2)

            ratios.append(alg.get_comp_ratio(opt))

    return eta1, eta2, ratios


def quality_of_ftp_just_additive_terms(prices, demands, num_repetitions, prediction_type="normal deviation"):
    pred_opt_off = pred.opt_off(prices, demands)
    phi = np.max(prices)
    input_length = len(prices)

    eta1 = []
    eta2 = []
    additive_terms = []

    for i in range(num_repetitions):
        alg_list, opt = create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands)

        for alg in alg_list:
            alg.run_full()

            e1 = alg.get_stock_error(opt)
            eta1.append(e1)

            e2 = alg.get_purchase_error(opt)
            eta2.append(e2)

            additive_terms.append(alg.get_additive_term(opt))

    return eta1, eta2, additive_terms


def quality_ftp_depending_on_phi(errors, ratios, phis):
    df = pd.DataFrame({"Phi": phis, "Errors": errors, "Ratios": ratios})
    phi_values = df["Phi"].drop_duplicates().tolist()  # get list of phi values
    data_per_phi = []

    for phi in phi_values:
        df_phi = df.loc[df['Phi'] == phi]
        errs = df_phi['Errors']
        rats = df_phi['Ratios']
        e, r = off.get_error_and_ratios(errs, rats, 3, num_repetitions=1, drop_duplicates=True)
        data_per_phi.append((phi, e, r))

    return data_per_phi


def evaluate_error_ratio(eval_func, treatments_input, treatments_predictions, num_generations, num_repetitions,
                         input_len, phi, file_name):
    for ti in treatments_input:
        for tp in treatments_predictions:
            eta1 = []
            eta2 = []
            ratios = []

            for i in range(num_generations):
                prices, demands = generate_instance(ti, input_len, phi)

                if tp == "uniform":
                    e1, e2, r = eval_func(prices, demands, num_repetitions * 10, prediction_type=tp)
                else:
                    e1, e2, r = eval_func(prices, demands, num_repetitions, prediction_type=tp)

                eta1.extend(e1)
                eta2.extend(e2)
                ratios.extend(r)

            ph.save_objects([eta1, eta2, ratios], f'Instances/{file_name}_input-{ti}_preds-{tp}_phi-{phi}.pkl')


def evaluate_phi_ratio(eval_func, treatments_input, treatments_predictions, num_generations, num_repetitions,
                       input_len, phi_list, file_name='evaluate-phi-ratio'):
    for ti in treatments_input:
        for tp in treatments_predictions:
            phi_errors_ratios = []
            for phi in tqdm(phi_list):
                eta1 = []
                eta2 = []
                ratios = []

                for i in range(num_generations):
                    prices, demands = generate_instance(ti, input_len, phi)

                    if tp == "uniform":
                        e1, e2, r = eval_func(prices, demands, num_repetitions * 10, prediction_type=tp)
                    else:
                        e1, e2, r = eval_func(prices, demands, num_repetitions, prediction_type=tp)

                    eta1.extend(e1)
                    eta2.extend(e2)
                    ratios.extend(r)
                phi_errors_ratios.append((phi, eta1, eta2, ratios))

            ph.save_object([phi_errors_ratios], f'Instances/{file_name}_input-{ti}_preds-{tp}.pkl')


def create_algs_for_one_repetition(i, pred_opt_off, prediction_type, phi, prices, demands):
    rng = default_rng(i)

    input_length = len(prices)

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

    return alg_list, opt
