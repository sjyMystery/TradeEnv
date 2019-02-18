import tensorflow as tf


class BasicCommission:
    def __init__(self):
        pass

    def calculate(self, quantity: tf.Tensor):
        raise NotImplementedError()
