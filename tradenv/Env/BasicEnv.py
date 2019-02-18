import numpy as np
import abc


class BasicEnv:
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
            Reset the environment and return the initial state
        :return:
            initial_state : 1d-np.array
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action: np.array):
        """
            Step into the environment and observe.
        :param action: np.array
        :returns:
            - new_state:  1d-np.array
            - reward: float
            - done: bool
        """
        raise NotImplementedError()

    @property
    def action_space(self):
        """

        :return: tradenv.Space.Space Object
        """
        raise NotImplementedError()

    @property
    def observe_space(self):
        """

        :return: tradenv.Space.Space Object
        """
        raise NotImplementedError()
