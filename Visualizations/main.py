import cost_barplots_from_df as barplots
import Simulations.history as history
from Simulations.instances import *

# Basic Settings
# Set the name of the CSV to use:
input = 'worst'

phi = 2
m = 3
n_cycles = 4

with_title = False
with_min_det = True

barplots.plot_cost_bars(phi, m, n_cycles, input, with_title, with_min_det)
