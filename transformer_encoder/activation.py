import numpy as np
from numpy.typing import NDArray


class Activation:
    # compute activation
    @staticmethod
    def forward(Z: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class Linear(Activation):
    @staticmethod
    def forward(Z: NDArray[np.float64]):
        return Z

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float64]):
        return 1


class ReLu(Activation):
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def backward(Z):
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z


class Softmax(Activation):
    @staticmethod
    def forward(Z):
        exp = np.exp(Z)
        return exp / exp.sum(0)

    @staticmethod
    def backward(Z):
        # derivative of softmax is included in loss function
        # to reduce time and space complexity
        return 1
