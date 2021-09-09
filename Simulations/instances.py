from Simulations.algorithms import *
import numpy as np


class RPA(Algorithm):
    """
    Implementation of the "Reservationspreis"-algorithm with parameter sqrt(phi).
    """
    def buy(self, phi, price, demand):
        if price <= np.sqrt(phi):
            self.x = self.get_max_amount_buy(demand)
        else:
            self.x = 0


class FtP(AlgorithmPred):
    """
    Implementation of the Follow-the-Prediction algorithm.
    """
    def buy(self, phi, price, demand):
        self.update_prediction()
        self.x = np.max([0, self.prediction - self.stock + demand])
