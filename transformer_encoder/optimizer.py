import numpy as np
from typing import Tuple
from numpy.typing import NDArray


class GradientDescent():
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def get_optimizer(self, size):
        return GradientDescent(self.alpha)
    
    def update_weights(self, dW, W):
        return W - self.alpha * dW

    def update_params(self, dW, db, W, b):
        W = self.update_weights(dW, W)
        b = b - self.alpha * db

        return W, b