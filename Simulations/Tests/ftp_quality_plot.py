import Simulations.offline as off

import pickle
import matplotlib.pyplot as plt

treatments_input = ['uniform']  # ['normal', 'uniform']
treatments_predictions = ['normal', 'uniform', '0', '1']

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
            