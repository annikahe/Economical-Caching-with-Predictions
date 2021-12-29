import Simulations.offline as off

import pickle
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    treatments_input = ['normal', 'uniform']  # ['normal', 'uniform']
    treatments_predictions = ['normal', 'uniform', '0', '1']

    treatment = 'cr'  # choose from ['cr', 'at'] ('cr' = competitive ratio, 'at' = additive term)

    for ti in treatments_input:
        for tp in treatments_predictions:
            with open(f'Instances/FtP-quality-phi-ratio-additive-terms_input-{ti}_preds-{tp}.pkl', 'rb') as inp:
                phi_errors_ratios = pickle.load(inp)
                phi_errors_ratios_df = pd.DataFrame(phi_errors_ratios)
                from collections import defaultdict
                d = defaultdict(list)
                for tup in phi_errors_ratios:
                    d[tup[0]].append(tup[1])
                    d[tup[0]].append(tup[2])
                    d[tup[0]].append(tup[3])
                phi_errors_ratios_df2 = pd.DataFrame(d)
                print(len(phi_errors_ratios))
                print(phi_errors_ratios_df)
                print(phi_errors_ratios_df2)

                if tp == "0" or tp == "1":
                    for tup in phi_errors_ratios:
                        phi = tup[0]
                        errors = tup[1]
                        ratios = tup[2]

                    e1, at1 = off.get_error_and_ratios(eta1, additive_terms, 15, 3)
                    e2, at2 = off.get_error_and_ratios(eta2, additive_terms, 15, 3)
                else:
                    e1, at1 = off.get_error_and_ratios(eta1, additive_terms, 50, 10)
                    e2, at2 = off.get_error_and_ratios(eta2, additive_terms, 50, 10)

                fig, ax = plt.subplots()
                # Plot theoretical competitive ratio

                ax.plot(e1, at1, label="Stock level based error")
                ax.plot(e2, at2, label="Purchase amount based error")
                ax.set_title(f"Input: {ti}, Predictions: {tp}")
                ax.set_xlabel("Error $\eta$")
                ax.set_ylabel("Additive Term")
                # ax.set_ylim([1, 2])
                at1 = [additive_term_stock_level(phi, eta) for eta in e1]
                at2 = [additive_term_purchase(phi, eta) for eta in e2]
                ax.plot(e1, at1, '--', color='tab:blue', label="Theoretical guarantee - stock level based error")
                ax.plot(e2, at2, '--', color='tab:orange', label="Theoretical guarantee - purchase amount based error")
                ax.grid(color='k', linestyle='dotted')
                if ti == "uniform" and tp == "normal":
                    ax.legend()
                plt.show()