import numpy as np
from numpy.typing import NDArray

# module imports
from transformer_encoder.activation import Softmax


class CategoricalCrossEntropy():
    @staticmethod
    def forward(Y, Y_hat):
        return -np.sum(Y * np.log(Y_hat))
        # return np.mean(-np.sum(Y * np.log(Y_hat), axis=-1))

    @staticmethod
    def backward(Y, Y_hat, activation, Z):
        if activation == Softmax:
            return Y_hat - Y
        else:
            raise ValueError("Categorical Cross Entropy is specifically designed for Softmax.")