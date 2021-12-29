import pickle
import matplotlib.pyplot as plt
import numpy as np

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

with open(f'../Tests/Instances/crude_oil_mindets.pkl', 'rb') as inp:
    mindets = pickle.load(inp)
    print(mindets)

    fig, ax = plt.subplots()

    x = [s for (s, _) in mindets]
    y = [mindet.mindet_history.alg.cost for (_, mindet) in mindets]

    print(y)

    ax.plot(x, y, label=alg_name_dict["mindet"])

    with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp2:
        algs = pickle.load(inp2)
        for alg_name in algs:
            if alg_name != "mindet" and alg_name != "ftp" and alg_name != "minrand":
                alg = algs[alg_name]
                ax.plot(x, np.full(len(x), alg.alg.cost), label=alg_name_dict[alg_name])

    with open(f'../Tests/Instances/crude_oil_ftps.pkl', 'rb') as inp3:
        ftps = pickle.load(inp3)
        ax.plot(x, [ftp.alg.cost for (_, ftp) in ftps], label=alg_name_dict["ftp"])

    with open(f'../Tests/Instances/crude_oil_minrands.pkl', 'rb') as inp4:
        minrands = pickle.load(inp4)
        ax.plot(x, [minrand.minrand_history.alg.cost for (_, minrand) in minrands], label=alg_name_dict["minrand"])

    ax.set_xscale("log")
    ax.set_xlabel("Sigma $\sigma$")
    ax.set_ylabel("Cost")
    ax.legend()
    plt.show()

