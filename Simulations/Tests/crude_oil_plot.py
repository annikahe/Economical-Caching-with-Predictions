import pickle

off.plot_ratios(eta1, ratios, 1, 1)
eta_norm = np.linspace(0, 0.1, 100)
y_opt = [off.comp_ratio_FtP_stock_error(phi, opt.cost, x) for x in eta_norm]
plt.plot(eta_norm, y_opt)
plt.show()

off.plot_ratios(eta2, ratios, 1, 1)
plt.show()