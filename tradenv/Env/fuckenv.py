import numpy as np
from tradenv.Env.BasicEnv import BasicEnv
from tradenv.Space import Box, Space

INIT_CASH = 0


class fuckenv(BasicEnv):
    def __init__(self):
        super(fuckenv, self).__init__()
        self.__cash = INIT_CASH
        self.has = False
        self.__action_space = Box(0, 1, (3,), )
        self.__last_in = 0
        self.__last_price = 0
        self.i = 0
        self.win = 0
        self.loss = 0
        """
            0 for do nothing
            1 for buy in 
            2 for sell out
        """
        self.__observe_space = Space(shape=(3,))
        """
            shape:
            0 current_PRICE
            1 has thing. 
            2 for last buy price 
        """

        self.total = 0
        self.get = 0

    def step(self, act):

        reward = 0

        if act == 1:
            if not self.has:
                self.has = True
                self.__last_in = self.__last_price
        if act == 2:
            if self.has:
                self.has = False
                reward = (self.__last_price) - self.__last_in
                self.__cash += reward
                if reward > 0:
                    self.win += 1
                else:
                    self.loss += 1
        price = self.__last_price + 1
        # print(price,self.__last_price,self.__last_in)
        self.__last_price = price

        self.i += 1

        if self.i >= 1000:
            self.total += 1
            self.get += self.equity

            a = abs(self.equity)
            print('%d\t %.3f\t' % (a, self.get / self.total))

        return np.array([price, self.has, self.__last_in]), reward, self.i >= 1000

    def reset(self):
        self.__cash = INIT_CASH
        self.__last_in = 0
        self.__last_price = 0
        self.i = 0
        # self.win = 0
        # self.loss = 0
        self.has = False
        return np.array([0, 0, 0])

    @property
    def win_rate(self):
        return float(self.win) / (self.win + self.loss) if self.win + self.loss > 0 else None

    @property
    def equity(self):
        return self.__cash

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observe_space
