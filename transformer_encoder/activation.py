import numpy as np
from numpy.typing import NDArray


"""
    This file features different activation functions that 
    introduces non-linearity into the model thus allowing it to 
    learn complex patterns. You can add other activation functions
    below by implementing the 'Activation' interface.
"""


class Activation:
    # compute activation
    @staticmethod
    def forward(Z: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float32]) -> NDArray[np.float32]:
        pass


class Linear(Activation):
    @staticmethod
    def forward(Z: NDArray[np.float32]):
        return Z

    # compute activation gradient
    @staticmethod
    def backward(Z: NDArray[np.float32]):
        return 1


"""
    ReLU stands for Rectified Linear Unit is a non-linear activation function 
    that outputs the input directly if it is positive, and zero otherwise.
    For reference of its formula and derivative check out 
    https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
"""


class ReLu(Activation):
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def backward(Z):
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z
    
    
class GeLu(Activation):
    @staticmethod
    def forward(Z):
        return 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * Z**3)))
    

"""
    Softmax is a non-linear activation function used in the last layer
    of a neural network for multiclass classification problems.
    For reference of its formula and derivative check out 
    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
"""


class Softmax(Activation):
    @staticmethod
    def forward(Z):
        # exp = np.exp(Z)
        exp = np.exp(Z - np.max(Z))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def backward(Z):
        # derivative of softmax is included in loss function
        # to reduce time and space complexity
        return 1
