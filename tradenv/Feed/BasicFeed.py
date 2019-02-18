import numpy as np


class BasicFeed:
    def __init__(self):
        self.__i = 0
        self.__data = np.empty(shape=(0, 9), dtype=np.float)

    def reset(self):
        self.__i = 0
        return self.next

    @property
    def next(self):
        current = self.__data[self.__i]
        self.__i += 1
        return current

    @property
    def done(self):
        return self.__i == self.__data.shape[0]

    @property
    def empty(self):
        return self.__data.shape[0] == 0

    @property
    def pos(self):
        return self.__i

    @property
    def data(self):
        return self.__data

    def append(self, array: np.array):
        assert len(array.shape) == 2 and array.shape[1] == 9, 'append data shape should be (None,9)'
        self.__data = np.append(self.__data, array,axis=0)
