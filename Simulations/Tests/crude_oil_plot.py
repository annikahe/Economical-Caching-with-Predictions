import pickle

with open(f'../Tests/Instances/crude_oil.pkl', 'rb') as inp:
    mindet = pickle.load(inp)
    print(mindet.mindet_history.alg.cost)

    opt_off = pickle.load(inp)
    print(opt_off.alg.cost)

# off.plot_ratios(eta1, ratios, 1, 1)
# eta_norm = np.linspace(0, 0.1, 100)
# y_opt = [off.comp_ratio_FtP_stock_error(phi, opt.cost, x) for x in eta_norm]
# plt.plot(eta_norm, y_opt)
# plt.show()
#
# off.plot_ratios(eta2, ratios, 1, 1)
# plt.show()