import numpy as np
from numpy.typing import NDArray
from typing import Tuple


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
        dW = dY.dot(X.T)  # derivative of weights
        db = np.sum(dY, axis=1)  # derivative of biases
        dX = W.T.dot(dY)  # derivative of input

        return dW, db, dX
