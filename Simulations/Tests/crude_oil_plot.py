import pickle

with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp:
    algs = pickle.load(inp)
    for alg_name in algs:
        alg = algs[alg_name]
        if alg_name == "off":
            print(f"$\cost(OFF) =$ {alg.alg.cost}")
        elif alg_name == "mindet":
            print("$\cost(MIN^{det}) =$" + f" {alg.mindet_history.alg.cost}")
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
