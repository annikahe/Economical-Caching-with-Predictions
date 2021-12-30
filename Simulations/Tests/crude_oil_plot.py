import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

alg_name_dict = {"off": "OFF", "ftp": "FtP", "mindet": "$MIN^{det}$",
                 "minrand": "$MIN^{rand}$", "rpa": "RPA", "threat": "Threat"}

with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp:
    algs = pickle.load(inp)
    print(algs)
    for alg_name in algs:
        print(alg_name)
        alg = algs[alg_name]
        if alg_name == "off":
            print(f"$\cost(OFF) =$ {alg.alg.cost}")
        elif alg_name == "ftp":
            print(f"$\cost(FtP) =$ {alg.alg.cost}")
        elif alg_name == "mindet":
            print("$\cost(MIN^{det}) =$" + f" {alg.mindet_history.alg.cost}")
        elif alg_name == "minrand":
            print("$\cost(MIN^{rand}) =$" + f" {alg.minrand_history.alg.cost}")
        elif alg_name == "rpa":
            print("$\cost(RPA) =$" + f" {alg.alg.cost}")
        elif alg_name == "threat":
            print("$\cost(Threat) =$" + f" {alg.alg.cost}")

# off.plot_ratios(eta1, ratios, 1, 1)
# eta_norm = np.linspace(0, 0.1, 100)
# y_opt = [off.comp_ratio_FtP_stock_error(phi, opt.cost, x) for x in eta_norm]
# plt.plot(eta_norm, y_opt)
# plt.show()
#
# off.plot_ratios(eta2, ratios, 1, 1)
# plt.show()

for a in ['rpa', 'threat']:
    with open(f'../Tests/Instances/crude_oil_mindets_{a}.pkl', 'rb') as inp:
        mindets = pickle.load(inp)

        fig, ax = plt.subplots()

        x = [s for (s, _) in mindets]
        y = [mindet.mindet_history.alg.cost for (_, mindet) in mindets]

        ax.plot(x, y, label=alg_name_dict["mindet"] + f"({alg_name_dict[a]})")

        with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp2:
            algs = pickle.load(inp2)
            for alg_name in algs:
                if alg_name != "mindet" and alg_name != "ftp" and alg_name != "minrand":
                    alg = algs[alg_name]
                    ax.plot(x, np.full(len(x), alg.alg.cost), label=alg_name_dict[alg_name])

        with open(f'../Tests/Instances/crude_oil_ftps.pkl', 'rb') as inp3:
            ftps = pickle.load(inp3)
            ax.plot(x, [ftp.alg.cost for (_, ftp) in ftps], label=alg_name_dict["ftp"])

        with open(f'../Tests/Instances/crude_oil_minrands_{a}.pkl', 'rb') as inp4:
            minrands = pickle.load(inp4)
            ax.plot(x, [minrand.minrand_history.alg.cost for (_, minrand) in minrands],
                    label=alg_name_dict["minrand"] + f"({alg_name_dict[a]})")

        ax.set_xscale("log")
        ax.set_xlabel("Sigma $\sigma$")
        ax.set_ylabel("Cost")
        ax.grid(color='k', linestyle='dotted')
        ax.legend()
        plt.show()


# two different comparisons (1. combine with RPA, 2. combine with Threat)

for a in ['rpa', 'threat']:
    with open(f'../Tests/Instances/crude_oil_mindets_{a}.pkl', 'rb') as inp:
        mindets = pickle.load(inp)

        with open(f'../Tests/Instances/crude_oil_opt_off.pkl', 'rb') as inp2:
            off = pickle.load(inp2)

        mindets_df = pd.DataFrame({"Costs": [m[1].mindet_history.get_cost() for m in mindets],
                                   "Errors": [m[1].algs_histories[1].get_purchase_error(off) for m in mindets]})

        mindets_df = mindets_df.sort_values(by=['Errors'])

        fig, ax = plt.subplots()

        x = mindets_df['Errors']
        y = mindets_df['Costs']

        ax.plot(x, y, label=alg_name_dict["mindet"] + f"({alg_name_dict[a]})")

        with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp2:
            algs = pickle.load(inp2)
            for alg_name in algs:
                if alg_name != "mindet" and alg_name != "ftp" and alg_name != "minrand":
                    alg = algs[alg_name]
                    ax.plot(x, np.full(len(x), alg.alg.cost), label=alg_name_dict[alg_name])

        with open(f'../Tests/Instances/crude_oil_ftps.pkl', 'rb') as inp3:
            ftps = pickle.load(inp3)

            ftps_df = pd.DataFrame({"Costs": [f[1].get_cost() for f in ftps],
                                    "Errors": [f[1].get_purchase_error(off) for f in ftps]})

            ftps_df = ftps_df.sort_values(by=['Errors'])

            ax.plot(ftps_df['Errors'], ftps_df['Costs'], label=alg_name_dict["ftp"])

        with open(f'../Tests/Instances/crude_oil_minrands_{a}.pkl', 'rb') as inp4:
            minrands = pickle.load(inp4)

            minrands_df = pd.DataFrame({"Costs": [m[1].minrand_history.get_cost() for m in minrands],
                                        "Errors": [m[1].algs_histories[1].get_purchase_error(off) for m in minrands]})

            minrands_df = minrands_df.sort_values(by=['Errors'])

            ax.plot(minrands_df['Errors'], minrands_df['Costs'], label=alg_name_dict["minrand"] + f"({alg_name_dict[a]})")

        ax.set_xlabel("Error $\eta$")
        ax.set_ylabel("Cost")
        ax.grid(color='k', linestyle='dotted')
        ax.legend()
        plt.show()


with open(f'../Tests/Instances/crude_oil_minrands_threat_different_eps.pkl', 'rb') as inp:
    minrands = pickle.load(inp)

    fig, ax = plt.subplots()

    with open(f'../Tests/Instances/crude_oil_opt_off.pkl', 'rb') as inp2:
        off = pickle.load(inp2)

    minrands_df = pd.DataFrame({"Eps": [m[0] for m in minrands],
                                "Costs": [m[2].minrand_history.get_cost() for m in minrands],
                                "Errors": [m[2].algs_histories[1].get_purchase_error(off) for m in minrands]})

    N = 7
    colors = plt.cm.Purples(np.linspace(0, 1, N))
    i = 2

    for eps in minrands_df["Eps"].drop_duplicates():

        if eps in [.1, .4, .7, .9]:
            minrands_eps_df = minrands_df[minrands_df["Eps"] == eps].sort_values(by=['Errors'])

            ax.plot(minrands_eps_df['Errors'], minrands_eps_df['Costs'],
                    label=f"$ \epsilon = {eps}$", color=colors[i])

            i += 1

    x = minrands_eps_df['Errors']

    with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp2:
        algs = pickle.load(inp2)
        for alg_name in algs:
            if alg_name != "mindet" and alg_name != "ftp" and alg_name != "minrand" and alg_name != "rpa":
                alg = algs[alg_name]
                ax.plot(x, np.full(len(x), alg.alg.cost), label=alg_name_dict[alg_name])

    with open(f'../Tests/Instances/crude_oil_ftps.pkl', 'rb') as inp3:
        ftps = pickle.load(inp3)

        ftps_df = pd.DataFrame({"Costs": [f[1].get_cost() for f in ftps],
                                "Errors": [f[1].get_purchase_error(off) for f in ftps]})

        ftps_df = ftps_df.sort_values(by=['Errors'])

        ax.plot(ftps_df['Errors'], ftps_df['Costs'], label=alg_name_dict["ftp"])

    ax.set_xlabel("Error $\eta$")
    ax.set_ylabel("Cost")
    ax.grid(color='k', linestyle='dotted')
    ax.legend()
    plt.show()

