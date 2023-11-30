import numpy as np
from numpy.typing import NDArray
from typing import Tuple

"""
    This file features common functions used in neural network 
    which, usually are matrix operations. You can add other common 
    neural network operations below given that you also provide the
    derivative of that operation.
"""


"""
    A common class used in performing an element-wise
    dot product of two vectors including its derivative
    calculation.
"""


class MatMul:
    @staticmethod
    def forward(X: NDArray[np.float64],
                Y: NDArray[np.float64]) -> NDArray[np.float64]:
        return X.dot(Y)

    @staticmethod
    def backward(dY: NDArray[np.float64],
                 X: NDArray[np.float64],
                 Y: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        dX = dY.dot(Y.T)  # derivative of x
        dY = X.T.dot(dY)  # derivative of y

        return dX, dY


"""
    A common class to calculate linear activation of neurons
    before passing it to an activation function. This operation includes 
    an element-wise dot product and adding biases to weight vector.
    This also includes derivative calculation of the linear activation.
"""


class PreActivation:
    @staticmethod
    def forward(X: NDArray[np.float64],
                W: NDArray[np.float64],
                b: NDArray[np.float64]) -> NDArray[np.float64]:
        b = np.reshape(b, (len(b), 1))
        return W.dot(X) + b

    @staticmethod
    def backward(dY: NDArray[np.float64],
                 X: NDArray[np.float64],
                 W: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # dW = 1 / batch_size * dY.dot(X.T)  # derivative of weights
        # db = 1 / batch_size * np.sum(dY, axis=1)  # derivative of biases
        dW = dY.dot(X.T)  # derivative of weights
        db = np.sum(dY, axis=1)  # derivative of biases
        dX = W.T.dot(dY)  # derivative of input

        return dW, db, dX
