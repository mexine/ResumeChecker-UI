import numpy as np

def glorot_normal(shape):
    """
    Glorot (Xavier) initialization for weights with a normal distribution.

    Parameters:
    - shape: Tuple, shape of the weight matrix.

    Returns:
    - weights: NumPy array, initialized weights.
    """
    fan_avg = (shape[0] + shape[1]) / 2.0
    std_dev = np.sqrt(2.0 / fan_avg)
    weights = np.random.normal(loc=0.0, scale=std_dev, size=shape)
    return weights