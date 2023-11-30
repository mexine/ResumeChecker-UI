import numpy as np
import abc
from typing import Tuple
from numpy.typing import NDArray


"""
    This file features different optimization algorithms to update 
    the model's parameters. You can add other optimization 
    algorithms below by extending the 'Optimizer' class.
"""


class Optimizer(abc.ABC):
    # initialize cache for optimizers that uses past gradients
    @abc.abstractmethod
    def get_optimizer(self, size: Tuple[int, int]):
        pass

    # updates weights and biases
    @abc.abstractmethod
    def update_params(self,
                      dW: NDArray[np.float64],  # weight gradients
                      db: NDArray[np.float64],  # bias gradients
                      W: NDArray[np.float64],  # current weights
                      b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:  # current biases
        pass


"""
    Gradient Descent is an iterative algorithm 
    to iteratively update the parameters (weights and biases) 
    of the model in a way that reduces the loss. This is achieved
    by subtracting the product of gradient/derivative and learning rate
    to the current parameters (weights and biases).
    For further details you can refer to 
    https://www.analyticsvidhya.com/blog/2021/03/understanding-gradient-descent-algorithm/
"""


class GradientDescent(Optimizer):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def get_optimizer(self, size):
        return GradientDescent(self.alpha)
    
    def update_weights(self, dW, W):
        return W - self.alpha * dW

    def update_params(self, dW, db, W, b):
        W = self.update_weights(dW, W)
        b = b - self.alpha * db

        return W, b


"""
    Adam combines ideas from both the AdaGrad and RMSProp optimizers to provide 
    adaptive learning rates for model parameters. Adam uses two moving averages 
    to adaptively update the learning rates for each parameter.
    For further details please check out
    https://towardsdatascience.com/learning-parameters-part-5-65a2f3583f7d
"""


class Adam(Optimizer):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, size=(0, 0)):
        self.m_dw, self.v_dw = np.zeros(size), np.zeros(size)
        self.m_db, self.v_db = np.zeros(size[0]), np.zeros(size[0])
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1

    def get_optimizer(self, size):
        return Adam(self.eta, self.beta1, self.beta2, self.epsilon, size)

    def update_weights(self, dw, W, add_t = True):
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)
        if (add_t): 
            self.t += 1

        W = W - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        return W

    def update_biases(self, db, b):
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        m_db_corr = self.m_db / (1 - self.beta1 ** self.t)
        v_db_corr = self.v_db / (1 - self.beta2 ** self.t)

        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
        return b

    def update_params(self, dw, db, W, b):
        W = self.update_weights(dw, W, add_t=False)
        b = self.update_biases(db, b)
        self.t += 1
        return W, b
