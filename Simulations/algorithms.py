class Algorithm:
    """
    The basic online algorithm has a stock and a cost accumulated so far.
    """
    def __init__(self, stock, cost):
        self.stock = stock
        self.cost = cost
        self.x = 0

    def run(self, gamma, phi, price, consumption):
        self.buy(phi, price, consumption)
        self.update_stock(consumption)
        self.update_cost(price)

    def buy(self, phi, price, consumption):
        """
        Place holder function for implementations in subclasses.
        """
        return 0

    def update_stock(self, consumption):
        if self.x + self.stock < consumption:
            print(f"The demand could not be covered. x too small. {self.x} + {self.stock} < {consumption}")
        elif self.x < consumption:
            self.stock -= (consumption - self.x)
        else:
            self.stock += (self.x - consumption)

    def update_cost(self, price):
        self.cost += price * self.x

    def update_turn(self, turn):
        pass

    def get_max_amount_buy(self, consumption):
        return 1 - self.stock + consumption


class MinDet(Algorithm):
    """
    Class for MIN^det algorithm.
    """
    def __init__(self, stock, cost, alg_list):
        super().__init__(stock, cost)
        self.alg_list = alg_list
        self.cycle = 0
        self.current_alg = 0

    def run(self, gamma, phi, price, consumption):
        """
        This is the main function of the MIN^det algorithm.
        For a given input (one time step of the sequence) it decides how much to buy and updates all parameters.
        It runs the helper functions defined below.
        """
        current_stocks_algs = [alg.stock for alg in self.alg_list]
        for i, alg in enumerate(self.alg_list):
            alg.run(gamma, phi, price, consumption)
        self.update_cycle(gamma, phi, price, current_stocks_algs)
        self.buy(phi, price, consumption)
        self.update_cost(price)  # defined in superclass
        self.update_stock(consumption)  # defined in superclass

    def buy(self, phi, price, consumption):
        self.x = self.alg_list[self.current_alg].x

    def update_cycle(self, gamma, phi, price, current_stocks_algs):
        stock_old = self.stock
        while (self.alg_list[self.current_alg]).cost > (self.cycle + 1) * gamma * phi:
            self.cycle += 1
            self.update_current_alg()
        stock_new = current_stocks_algs[self.current_alg]  # self.alg_list[self.current_alg].stock
        self.cost += max(stock_new - stock_old, 0) * price
        #print(f"Switching cost: max({stock_new} - {stock_old}, 0) * {price} = {max(stock_new - stock_old, 0) * price}")
        self.stock = stock_new

    def update_current_alg(self):
        self.current_alg = self.cycle % len(self.alg_list)


class AlgorithmPred(Algorithm):
    """
    This is the class of online algorithms with predictions.
    The predictions used here tell the algorithm how much to have in stock at least at the end of the current time step.
    """
    def __init__(self, stock, cost, predictions):
        super().__init__(stock, cost)
        self.prediction = 0
        self.remaining_predictions = predictions
        self.remaining_predictions.reverse()  # reverse list to be able to pop first element from list

    def update_prediction(self):
        self.prediction = self.remaining_predictions.pop()


class AlgorithmTurn(Algorithm):
    """
    This type of example algorithms also get the boolean variable `turn` as input.
    turn = 0 => Alg currently not in use by MIN^det
    turn = 1 => Alg currently in use by MIN^det
    """
    def __init__(self, stock, cost):
        super().__init__(stock, cost)
        self.turn = False

    def update_turn(self, turn):
        self.turn = turn


class MinDetTurn(AlgorithmTurn):
    """
    Class for MIN^det algorithm.
    """
    def __init__(self, stock, cost, alg_list):
        super().__init__(stock, cost)
        self.alg_list = alg_list
        self.cycle = 0
        self.current_alg = 0

    def run(self, gamma, phi, price, consumption):
        """
        This is the main function of the MIN^det algorithm.
        For a given input (one time step of the sequence) it decides how much to buy and updates all parameters.
        It runs the helper functions defined below.
        """
        self.update_cycle(gamma, phi, price)
        for i, alg in enumerate(self.alg_list):
            #print(i == self.current_alg)
            alg.update_turn(i == self.current_alg)
            alg.run(gamma, phi, price, consumption)
        self.buy(phi, price, consumption)
        self.update_cost(price)  # defined in superclass
        self.update_stock(consumption)  # defined in superclass

    def buy(self, phi, price, consumption):
        self.x = self.alg_list[self.current_alg].x

    def update_cycle(self, gamma, phi, price):
        while (self.alg_list[self.current_alg]).cost + price * self.alg_list[self.current_alg].x > (self.cycle + 1) * gamma * phi:
            self.cycle += 1
            self.update_current_alg()

    def update_current_alg(self):
        self.current_alg = self.cycle % len(self.alg_list)


class AlgorithmTurnPred(AlgorithmPred):
    def __init__(self, stock, cost, predictions):
        super().__init__(stock, cost, predictions)
        self.turn = False

    def update_turn(self, turn):
        self.turn = turn


class AlgorithmRandom(Algorithm):
    """
    This is the class of online algorithms for the EC problem with random purchase amounts.
    """

    def __init__(self, stock, cost):
        super().__init__(stock, cost)

    def buy(self, phi, price, consumption):
        import numpy as np

        lower_bound = np.max([0, consumption - self.stock])
        upper_bound = 1 - self.stock + consumption
        self.x = (upper_bound - lower_bound) * np.random.rand() + lower_bound
