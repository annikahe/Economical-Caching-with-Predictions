import Simulations.offline as off

import pickle
import matplotlib.pyplot as plt

treatments_input = ['normal', 'uniform']
treatments_predictions = ['normal', 'uniform', '0', '1']

for ti in treatments_input:
    for tp in treatments_predictions:
        with open(f'Instances/FtP_quality_input-{ti}_preds-{tp}.pkl', 'rb') as inp:
            eta1 = pickle.load(inp)
            print(eta1)

            eta2 = pickle.load(inp)
            print(eta2)

            ratios = pickle.load(inp)
            print(ratios)

            if tp == "0" or tp == "1":
                e, r = off.get_error_and_ratios(eta1, ratios, 15, 3)
            else:
                e, r = off.get_error_and_ratios(eta1, ratios, 50, 10)
            plt.plot(e, r)
            plt.title(f"Input: {ti}, Predictions: {tp}")
            plt.xlabel("Error $\eta$")
            plt.ylabel("Competitive Ratio")
            plt.show()
