from tradenv.Commission.BasicCommission import BasicCommission
import tensorflow as tf


class NoCommission(BasicCommission):
    def __init__(self):
        super(NoCommission, self).__init__()

        pass

    def calculate(self, quantity: tf.Tensor):
        each = tf.multiply(quantity, tf.constant(0))
        total = tf.cumsum(each)
        return total

