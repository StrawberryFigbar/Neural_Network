import numpy as np

import layer


class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_propagation(self, input):
        return np.reshape(input, (1, -1))

    def backward_propagation(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)
