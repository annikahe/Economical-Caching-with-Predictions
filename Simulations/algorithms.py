import numpy as np

class Algorithm:
    """
    Basic framework for online algorithms for the Economical Caching Problem.
    Precise online algorithms will be defined as subclasses of this class.
    Subclasses need to provide an implementation of the function buy(phi, price, demand).
    ...

    Attributes
    ----------
    stock : float
        Current stock level.
    cost : float
        Accumulated cost incurred by the algorithm so far.
    x : float
        Current purchase amount.
        x is a value > 0 that indicates how much the algorithm is going to buy in the current step.

    Methods
    -------
    run(gamma, phi, price, demand)
        Executes one step of the algorithm: Buys the amount x and updates the stock and accumulated cost accordingly.

    buy(phi, price, demand)
        Sets self.x. The function is not implemented in this class
        as it will be overwritten by the corresponding function in the subclasses.

    update_stock(demand)
        Updates the stock level of the algorithm based on the current demand and self.x.
        In the case that the initial stock level and the amount that we are going to buy is not sufficient
        to cover the current demand, it will print out a string stating this error.
        The buy-function should always be implemented in a way that this does not happen.

    update_cost(price)
        Adds the current cost (price times purchase amount) to the accumulated cost of the algorithm.

    update_turn(self, turn)
        A placeholder function for an artificial class of algorithms ("AlgorithmTurn")
        that know when they are followed by MIN^det.
        Might be removed in the future.

    get_max_amount_buy(self, demand)
        Returns the maximum amount that the algorithm can buy in this time step.
    """
    def __init__(self, stock, cost):
        self.stock = stock
        self.cost = cost
        self.x = 0

    def run(self, gamma, phi, price, demand):
        self.buy(phi, price, demand)
        self.update_stock(demand)
        self.update_cost(price)

    def buy(self, phi, price, demand):
        """
        Placeholder function for implementations in subclasses.
        """
        pass

    def update_stock(self, demand):
        if self.x + self.stock < demand:
            print(f"The demand could not be covered. x too small. {self.x} + {self.stock} < {demand}. Rounding up.")
            self.x = demand - self.stock
            self.stock = 0
        elif self.x < demand:
            self.stock -= (demand - self.x)
        else:
            self.stock += (self.x - demand)

    def update_cost(self, price):
        self.cost += price * self.x

    def update_turn(self, turn):
        pass

    def get_max_amount_buy(self, demand):
        return 1 - self.stock + demand


class MinDet(Algorithm):
    """
    Class for MIN^det algorithm.

    Attributes
    ----------
    alg_list : list
        List of the input algorithms that MIN^det uses.
    cycle : int
        Index of the current cycle (an cycle ends when MIN^det switches to another algorithm).
    current_alg : int
        Index of the input algorithm that is currently followed by MIN^det.

    Methods
    -------
    run(gamma, phi, price, demand)
        Overwrites the implementation of this function in the superclass.
        Additionally to just executing the algorithm itself, it also needs to execute the input algorithms
        and update self.cycle and self.current_alg.

    buy(phi, price, demand)
        Buys the same amount that the currently followed input algorithm buys.

    update_cycle(gamma, phi, price, current_stocks_algs)
        Update the cycle index if the accumulated cost of the currently followed input algorithm exceeds the threshold.

    update_current_alg()
        Update self.current_alg based on self.cycle.
    """
    def __init__(self, stock, cost, alg_list):
        super().__init__(stock, cost)
        self.alg_list = alg_list
        self.cycle = 0
        self.current_alg = 0

    def run(self, gamma, phi, price, demand):
        """
        This is the main function of the MIN^det algorithm.
        For a given input (one time step of the sequence) it decides how much to buy and updates all parameters.
        It runs the helper functions defined below.
        """
        current_stocks_algs = [alg.stock for alg in self.alg_list]
        for i, alg in enumerate(self.alg_list):
            alg.run(gamma, phi, price, demand)
        self.update_cycle(gamma, phi, price, current_stocks_algs)
        self.buy(phi, price, demand)
        self.update_cost(price)  # defined in superclass
        self.update_stock(demand)  # defined in superclass

    def buy(self, phi, price, demand):
        self.x = self.alg_list[self.current_alg].x

    def update_cycle(self, gamma, phi, price, current_stocks_algs):
        stock_old = self.stock
        while (self.alg_list[self.current_alg]).cost > (self.cycle + 1) * gamma * phi:
            self.cycle += 1
            self.update_current_alg()
        stock_new = current_stocks_algs[self.current_alg]  # self.alg_list[self.current_alg].stock
        self.cost += max(stock_new - stock_old, 0) * price
        self.stock = stock_new

    def update_current_alg(self):
        self.current_alg = self.cycle % len(self.alg_list)


class AlgorithmPred(Algorithm):
    """
    This is the class of online algorithms with predictions.
    The predictions used here tell the algorithm how much to have in stock at least at the end of the current time step.

    Attributes
    ----------
    prediction : float
        Prediction for the current time step.
    remaining_predictions : list
        List of predictions for the time steps remaining from the current time step on.

    Methods
    -------
    update_prediction()
        Gets the next entry from the list of the remaining predicitions and stores it in self.predicition.
        The value is removed from the list.

    """
    def __init__(self, stock, cost, predictions):
        super().__init__(stock, cost)
        self.prediction = 0
        self.remaining_predictions = predictions
        self.remaining_predictions.reverse()  # reverse the list to be able to pop first element from list

    def update_prediction(self):
        self.prediction = self.remaining_predictions.pop()


class AlgorithmRandom(Algorithm):
    """
    This is the class of online algorithms for the EC problem with uniformly random purchase amounts.

    Attributes
    ----------
    No additional attributes besides the attributes defined in the super class Algorithm.

    Methods
    -------
    buy(phi, price, demand)
        Sets self.x to a value uniformly drawn from the interval that is lower bounded by
        what the algorithm needs to buy at least in this step and what it can buy at most.

    """

    def __init__(self, stock, cost):
        super().__init__(stock, cost)

    def buy(self, phi, price, demand):
        import numpy as np

        lower_bound = np.max([0, demand - self.stock])
        upper_bound = 1 - self.stock + demand
        self.x = (upper_bound - lower_bound) * np.random.rand() + lower_bound


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
