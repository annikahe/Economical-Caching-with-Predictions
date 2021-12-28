import numpy as np

import Simulations.offline as off

import pickle
import matplotlib.pyplot as plt


def additive_term_stock_level(phi, eta):
    return np.min([phi * eta, (phi - 1)*(eta - 1) + phi])


def additive_term_prediction(phi, eta):
    return np.min([phi * eta, (((phi - 1) * (eta - 1)) / 2) + phi])


treatments_input = ['uniform']  # ['normal', 'uniform']
treatments_predictions = ['normal', 'uniform', '0', '1']

treatment = 'at'  # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

phi = 100

if treatment == 'cr':
    for ti in treatments_input:
        for tp in treatments_predictions:
            with open(f'Instances/FtP_quality_input-{ti}_preds-{tp}.pkl', 'rb') as inp:
                eta1 = pickle.load(inp)
                eta2 = pickle.load(inp)
                ratios = pickle.load(inp)

                if tp == "0" or tp == "1":
                    e1, r1 = off.get_error_and_ratios(eta1, ratios, 15, 3)
                    e2, r2 = off.get_error_and_ratios(eta2, ratios, 15, 3)
                else:
                    e1, r1 = off.get_error_and_ratios(eta1, ratios, 50, 10)
                    e2, r2 = off.get_error_and_ratios(eta2, ratios, 50, 10)

                fig, ax = plt.subplots()
                # Plot theoretical competitive ratio

                ax.plot(e1, r1, label="Stock level based error")
                ax.plot(e2, r2, label="Prediction based error")
                ax.set_title(f"Input: {ti}, Predictions: {tp}")
                ax.set_xlabel("Error $\eta$")
                ax.set_ylabel("Competitive Ratio")
                ax.set_ylim([1, 2])
                ax.grid(color='k', linestyle='dotted')
                ax.legend()
                plt.show()

                ###

                fig, ax = plt.subplots()

                e1, r1 = off.get_error_and_ratios(eta1, ratios, 50, 10, drop_duplicates=False)
                e2, r2 = off.get_error_and_ratios(eta2, ratios, 50, 10, drop_duplicates=False)

                ax.plot(e1, e2)

                ax.grid(color='k', linestyle='dotted')
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                ax.set_xlabel("Stock level based error $\eta_1$")
                ax.set_ylabel("Prediction based error $\eta_2$")

                plt.show()

else:
    for ti in treatments_input:
        for tp in treatments_predictions:
            with open(f'Instances/FtP_quality_input_just_additive_terms-{ti}_preds-{tp}.pkl', 'rb') as inp:
                eta1 = pickle.load(inp)
                eta2 = pickle.load(inp)
                additive_terms = pickle.load(inp)

                if tp == "0" or tp == "1":
                    e1, at1 = off.get_error_and_ratios(eta1, additive_terms, 15, 3)
                    e2, at2 = off.get_error_and_ratios(eta2, additive_terms, 15, 3)
                else:
                    e1, at1 = off.get_error_and_ratios(eta1, additive_terms, 50, 10)
                    e2, at2 = off.get_error_and_ratios(eta2, additive_terms, 50, 10)

                fig, ax = plt.subplots()
                # Plot theoretical competitive ratio

                ax.plot(e1, at1, label="Stock level based error")
                ax.plot(e2, at2, label="Prediction based error")
                ax.set_title(f"Input: {ti}, Predictions: {tp}")
                ax.set_xlabel("Error $\eta$")
                ax.set_ylabel("Additive Term")
                # ax.set_ylim([1, 2])
                at1 = [additive_term_stock_level(phi, eta) for eta in e1]
                at2 = [additive_term_prediction(phi, eta) for eta in e2]
                ax.plot(e1, at1, '--', color='tab:blue', label="Theoretical guarantee - stock level based error")
                ax.plot(e2, at2, '--', color='tab:orange', label="Theoretical guarantee - prediction based error")
                ax.grid(color='k', linestyle='dotted')
                ax.legend()
                plt.show()

                ###

                fig, ax = plt.subplots()

                e1, at1 = off.get_error_and_ratios(eta1, additive_terms, 50, 10, drop_duplicates=False)
                e2, at2 = off.get_error_and_ratios(eta2, additive_terms, 50, 10, drop_duplicates=False)

                ax.plot(e1, e2)

                ax.grid(color='k', linestyle='dotted')
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                ax.set_xlabel("Stock level based error $\eta_1$")
                ax.set_ylabel("Prediction based error $\eta_2$")

                plt.show()

            ###

            # fig, ax = plt.subplots()
            #
            # e1, r1 = off.get_error_and_ratios(eta1, ratios, 50, 10, drop_duplicates=False)
            # e2, r2 = off.get_error_and_ratios(eta2, ratios, 50, 10, drop_duplicates=False)
            #
            # error_ratio = [e1[i] / e2[i] for i in range(len(e1)) if e2[i] != 0]
            # ax.plot(error_ratio)
            #
            # ax.grid(color='k', linestyle='dotted')
            # ax.set_xlabel("Stock level based error $\eta_1$")
            # ax.set_ylabel("Prediction based error $\eta_2$")
            #
            # plt.show()
            