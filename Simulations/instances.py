from Simulations.algorithms import *
import numpy as np


class AlgorithmEx1(AlgorithmTurn):
    """
    This class provides the implementation of Example Algorithm 1.
    This is specifically tailored to the input sequences
     prices = [1, phi, 1, phi, ...]
    demands = [0,  1,  0,  1,  ...].
    """
    def buy(self, phi, price, demand):
        if self.turn:
            if price == 1:  # and demand == 0
                self.x = 0
            else:  # if price == phi and demand == 1
                self.x = 1
        else:
            if price == 1:  # and demand == 0
                self.x = 1
            else:  # if price == 0 and demand == phi
                self.x = 0


class AlgorithmEx2(AlgorithmTurnPred):
    def buy(self, phi, price, demand):
        if self.turn:
            if price == 1:  # and demand == 0
                self.prediction = 0
            else:  # if price == phi and demand == 1
                self.prediction = 1
        else:
            if price == 1:  # and demand == 0
                self.prediction = 1
            else:  # if price == 0 and demand == phi
                self.prediction = 0
        self.x = np.max([0, self.prediction - self.stock + demand])


class RPA(Algorithm):
    """
    This class provides the implementation of the "Reservationspreis"-algorithm with parameter sqrt(phi).
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
