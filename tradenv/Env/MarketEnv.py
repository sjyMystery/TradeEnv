import numpy as np

from tradenv.Env.BasicEnv import BasicEnv
from tradenv.Space import Discrete, Space
from tradenv.Feed import BasicFeed


class MarketEnv(BasicEnv):
    def reset(self):
        self.__cash = self.__init_cash
        self.__holding = 0
        self.__last_in_price = 0
        self.__bar_state = self.__feed.reset()

        self.__win = 0
        self.__loss = 0

        return self.state_from_bar(self.__bar_state)

    @property
    def win_rate(self):
        return float(self.__win) / (self.__win + self.__loss) if self.__win + self.__loss > 0 else None

    def __mkt_in(self, next_state):
        self.__holding = int(self.__cash / next_state[0])
        self.__cash -= self.__holding * next_state[0]
        self.__last_in_price = next_state[0]

    def __mkt_out(self, next_state):
        out_price = next_state[4]
        self.__cash += self.__holding * next_state[4]
        self.__holding = 0

        if out_price > self.__last_in_price:
            self.__win += 1
        else:
            self.__loss += 1

        return out_price - self.__last_in_price

    @property
    def in_price(self):
        return self.__bar_state[3]

    @property
    def out_price(self):
        return self.__bar_state[7]

    def step(self, action: int):
        assert action in [0, 1, 2], 'Action Must Be In 0,1,2'
        assert self.__bar_state is not None, "You Must Reset The Environment First!"

        next_bar_state: np.array = self.__feed.next

        reward = 0

        if action is 1:
            if self.__holding is 0:
                """
                   either holding , or to do nothing.
                """
                self.__mkt_in(next_bar_state)
        elif action is 2:
            if self.__holding is not 0:
                profit = self.__mkt_out(next_bar_state)
                reward = profit if profit > 0 else -1
        else:
            pass
        self.__bar_state = next_bar_state
        return self.state_from_bar(self.__bar_state), reward, self.__feed.done

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observe_space(self):
        return self.__observe_space

    def __init__(self, cash, feed: BasicFeed, on_close=False):
        super(MarketEnv, self).__init__()

        self.__action_space = Discrete(3)
        """
            0 for do nothing
            1 for buy in 
            2 for sell out
        """

        self.__observe_space = Space(shape=(4,))
        """
            shape:
            0 AskOpen
            1 AskLow
            2 AskHigh
            3 AskClose
            4 BidOpen
            5 BidLow
            6 BidHigh
            7 BidClose
            8 DateTime in minutes (UTC offset)
            9 dim for holding? quantity
            10 dim for last buy price 
            11 dim for cash
        """

        self.__cash = self.__init_cash = cash
        self.__feed = feed
        self.__holding = 0
        self.__last_in_price = 0
        self.__bar_state = None
        self.__on_close = on_close

        self.__win = 0
        self.__loss = 0

    @property
    def cash(self):
        return self.__cash

    @property
    def win(self):
        return self.__win

    @property
    def loss(self):
        return self.__loss

    @property
    def equity(self):
        return self.__cash + self.out_price * self.__holding

    def state_from_bar(self, bar):
        return np.append(bar[[0,4]], [self.__holding > 0, self.__last_in_price])
