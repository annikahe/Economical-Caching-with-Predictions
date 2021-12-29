from Simulations.algorithms import *
import Simulations.offline as off
import Simulations.predictions as pred
from Simulations.history import *
import Simulations.pickle_helpers as ph

import numpy as np
import pandas as pd



# Prices
# crude oil prices_daily in dollars (02.01.1986 - 09.07.2018)
prices_daily = pd.read_excel(r'../../Data/Crude_Oil_Prices_Daily.xlsx')

quarters_starts = ['01-01', '04-01', '07-01', '10-01']
quarters_ends = ['03-31', '06-30', '09-30', '12-31']


def get_quarter_average(year, quarter_start, quarter_end):
    quarter = prices_daily[(prices_daily['Date'] >= f'{year}-{quarter_start}') & (prices_daily['Date'] <= f'{year}-{quarter_end}')]
    quarter = quarter[['Closing Value']].dropna()  # remove NaNs
    return quarter[['Closing Value']].sum()/len(quarter)


prices_quarterly_df = pd.DataFrame([get_quarter_average(year, quarters_starts[i], quarters_ends[i]) for year in range(1994, 2019) for i in range(4)])
prices_quarterly_df = prices_quarterly_df.dropna()

prices_quarterly = prices_quarterly_df['Closing Value'].values.tolist()

min_price = np.min(prices_quarterly)
prices_quarterly_normalized = prices_quarterly / min_price


# Demands
# crude oil demand (1994 Q1 - 2019 Q3)
demands_supply_df = pd.read_csv(r'../../Data/Supply_Demand_Oil.csv')
demands_supply_df = demands_supply_df[demands_supply_df['Quarter'] <= f'2018 Q4']

demands = demands_supply_df['Oced_D_Europe'].values.tolist()

max_demand = np.max(demands)
demands_normalized = demands / max_demand

prices = prices_quarterly_normalized
demands = demands_normalized

phi = np.max(prices)

# eta1, eta2, ratios = off.quality_of_FtP(prices[:10],
#                                         demands[:10],
#                                         [0, 0.000000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10, 100],
#                                         20)
#
# print(f"Max price = {prices[:10].max()}")
# print(f"Min price = {prices[:10].min()}")
# print(f"Max demand = {demands[:10].max()}")
# print(f"Min demand = {demands[:10].min()}")
#
# # optimal solution
# pred_opt_off = off.opt_stock_levels(prices[:10], demands[:10])
# opt = FtP(0, 0, pred_opt_off)
# for i in range(10):
#     opt.run(1, phi, prices[i], demands[i])

pred_opt_off = pred.opt_off(prices, demands)
opt_off = History(1, phi, prices, demands, FtP(0, 0, pred_opt_off))
opt_off.run_full()


ftp = FtP(0, 0, pred.predictions_normal_off(pred_opt_off))
mindet = MinDetHistory(1, phi, prices, demands, [RPA(0, 0), ftp])
mindet.run_full()


# rpa = History(1, phi, prices, demands, RPA(0, 0))
# rpa.run_full()
#
#
# threat = History(1, phi, prices, demands, Threat(0, 0))
# threat.run_full()


ph.save_object({"off": opt_off, "mindet": mindet}, '../Tests/Instances/crude_oil.pkl')
