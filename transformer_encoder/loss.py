import numpy as np
from numpy.typing import NDArray

# module imports
from transformer_encoder.activation import Activation, Softmax


"""
    This file features different loss functions that quantifies 
    the discrepancy between the model's predictions and the actual values. 
    Loss functions guides the learning process to improve the model's performance.
    You can add other loss functions below by implementing the 'Loss' interface.
    
    Also note that to reduce the time and space complexity of the model,
    we include the last activation function derivative to backpropagation function.
    Moreover, not all activation functions are compatible to all loss functions so,
    make sure to validate if an activation function fits with the loss function.
"""


class Loss:
    @staticmethod
    def forward(Y: NDArray[np.float32], Y_hat: NDArray[np.float32]) -> np.float32:
        pass

    @staticmethod
    def backward(Y: NDArray[np.float32],
                 Y_hat: NDArray[np.float32],
                 activation: Activation,
                 dZ: NDArray[np.float32]) -> NDArray[np.float32]:
        pass


"""
    Categorical Cross Entropy is used for multiclass classification problems.
    Categorical Cross Entropy is often used in scenarios where each example 
    belongs to one and only one class. The true target values are represented using one-hot encoding, 
    where the actual class is indicated by a 1 (true) and other classes are indicated by 0 (false).
    That is why, softmax activation function is the only applicable for categorical cross entropy.
    For further information about its formula and concept, check out
    https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
"""


class CategoricalCrossEntropy(Loss):
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