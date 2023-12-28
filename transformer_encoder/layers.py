import abc
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

# project imports
from transformer_encoder.loss import Loss
from transformer_encoder.activation import Activation, Softmax, Linear
from transformer_encoder.optimizers import Optimizer, GradientDescent
from transformer_encoder.common import PreActivation, MatMul
from transformer_encoder.initialization import glorot_normal


"""
    This file features different neural network layers.
    You can add additional layer types (e.g. Convolutional Layers,
    Recurrent Layers) by extending the 'Layer' class.
"""


class Layer(abc.ABC):
    """
        This method is used to reinitialize the chosen learning
        optimizer. We reinitialized it to provide the layer size
        because adaptive optimizers requires a cache of previous
        parameters.
    """
    @abc.abstractmethod
    def set_optimizer(self, optimizer: Optimizer) -> None:
        pass

    """
        This method is used for forward propagation. You need to 
        provide the X (input) with the shape of (input_size, sample_size).
        You can also specify if the model is still training or 
        already for production. This is helpful to prevent the model from 
        computing derivatives when the model is already intended for 
        production.
    """
    @abc.abstractmethod
    def forward(self,
                X: NDArray[np.float32],
                training: bool = True) -> NDArray[np.float32]:
        pass

    """
        This method is used for backpropagation. It takes the dY (derivatives
        from previous layers), batch size, Y (actual values), Y_hat (predicted
        values from the last layer), and the loss function.
        
        Notice that the last three arguments (Y, Y_hat, and loss function) is 
        used just for last layers. The computation of activation function derivative
        is sometimes included in loss function derivative to reduce the model's
        time and space complexity.
    """
    @abc.abstractmethod
    def backward(self,
                 dY: NDArray[np.float32],
                 Y: NDArray[np.float32] = None,
                 Y_hat: NDArray[np.float32] = None,
                 loss_function: Loss = None) -> NDArray[np.float32]:
        pass


"""
    The Dense layer, also referred as a Linear layer in most articles
    is the common layer type for feed forward networks.
    For additional explanation, you can check out
    https://www.analyticsvidhya.com/blog/2022/01/feedforward-neural-network-its-layers-functions-and-importance/
"""


class Dense(Layer):
    """
        :arg size [corresponds to a tuple (input_size, output_size). The input size is the output
                    dimension of the previous layer while the output size is the target output
                    dimension of the current layer. The activation]
        :arg activation [refers to the chosen activation function]
    """
    def __init__(self, size: Tuple[int, int], activation: Activation):
        # initialize parameters
        x_size, y_size = size
        self.W = glorot_normal((y_size, x_size))
        self.b = np.zeros((y_size,))
        self.activation = activation
        self.optimizer = GradientDescent()

        # initialize cache
        self.X = None
        self.Z = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer.get_optimizer(self.W.shape)

    def forward(self, X, training=True):
        # forward propagation
        Z = PreActivation.forward(X, self.W, self.b)
        A = self.activation.forward(Z)

        # cache values for gradient descent
        if training is True:
            self.X = X
            self.Z = Z

        # return activation
        return A

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        if loss_function is None or Y is None or Y_hat is None:
            dY = self.activation.backward(self.Z) * dY
        else:
            dY = loss_function.backward(Y, Y_hat, self.activation, self.Z)

        # calculate gradient
        dW, db, dX = PreActivation.backward(dY, self.X, self.W)

        # update parameters
        W, b = self.optimizer.update_params(dW, db, self.W, self.b)
        self.W = W
        self.b = b

        return dX  # return derivative of input
    
    def get_trainable_variables(self):
        return {
            'dense/weights': self.W, 
            'dense/bias': self.b
        }
    
    def set_trainable_variables(self, weights: NDArray[np.float32], bias: NDArray[np.float32]):
        self.W = weights
        self.b = bias

    def reinstantiate(self):
        y_size, x_size = self.W.shape
        return type(self)((x_size, y_size), self.activation)
    

"""
    Word Embedding layer maps the input token ids to their initial vector representations. 
"""


class WordEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim):
        self.W = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        self.X = None
        self.optimizer = GradientDescent()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer.get_optimizer(self.W.shape)

    def forward(self, X: NDArray[np.int64], training=True):
        self.X = X
        return self.W[X, :]

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        dWE = np.zeros_like(self.W)
        np.add.at(dWE.T, self.X, dY.T)

        # Update word embeddings using Adam optimizer
        self.W = self.optimizer.update_weights(dWE, self.W)

        # Return gradients with respect to the input (not needed for this layer)
        return None
    
    def get_trainable_variables(self):
        return {
            'word_embedding/weights': self.W, 
        }

    def set_trainable_variables(self, weights: NDArray[np.float32]):
        if (self.W.shape != weights.shape):
            raise ValueError(str(type(self)) + ': Wrong weights shape, expected ' + str(self.W.shape) + ', received ' + str(weights.shape))
        
        self.W = weights

    def reinstantiate(self):
        raise ValueError('WordEmbedding layers are typically not stacked.')


"""
    Positional Encoding layer is a technique used in transformer-based architectures to
    incorporate information about the position of words or tokens in a sequence into the model.
    For additional details please refer to https://research.google/pubs/pub46201/
"""


class PositionalEncoding(Layer):
    def __init__(self,
                 sequence_len: int,
                 dimension: int,
                 eta: int = 10000):
        temp = np.zeros((sequence_len, dimension), dtype=np.float32)  # initialize positional encodings
        for k in range(sequence_len):
            for i in np.arange(int(dimension / 2)):
                denominator = np.power(eta, 2 * i / dimension)
                temp[k, 2 * i] = np.sin(k / denominator)
                temp[k, 2 * i + 1] = np.cos(k / denominator)

        self.dimension = dimension
        self.pos_encoding = temp  # cache positional encodings

    def set_optimizer(self, optimizer):
        pass

    def get_positional_encoding(self):
        return self.pos_encoding

    def forward(self, X, training=True):
        seq_len = len(X)
        return X + self.pos_encoding[:seq_len, :]

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        return dY
    
    def get_trainable_variables(self):
        return {
            'positional_encoding': self.pos_encoding
        }

    def set_trainable_variables(self, pos_encoding):
        self.pos_encoding = pos_encoding
        # pass
    
    def reinstantiate(self):
        raise ValueError('PositionalEncoding layers are typically not stacked.')



"""
    Multi-Head Attention is a parrallelized version of the Self Attention Layer. 
    It has num_heads number of heads or parrallelized Self Attention Layer. 
"""

class MultiHeadAttention(Layer):
    def __init__(self, num_heads: int, dimension: int):
        self.num_heads = num_heads
        # self.sequence_len = sequence_len
        self.dimension = dimension
        self.key_dim = self.dimension // self.num_heads

        self.query = Dense((dimension, dimension), Linear())
        self.key = Dense((dimension, dimension), Linear())
        self.value = Dense((dimension, dimension), Linear())

        self.linear = Dense((dimension, dimension), Linear())

    def set_optimizer(self, optimizer):
        pass

    def forward(self, X, attention_mask, training=True):
        seq_len = len(X)

        # linear transformation for query, key, value
        query_val = self.query.forward(X)
        key_val = self.key.forward(X)
        value_val = self.value.forward(X)

        # split embeddings by num_heads
        query_val = np.reshape(query_val, (seq_len, self.num_heads, self.key_dim))
        key_val = np.reshape(key_val, (seq_len, self.num_heads, self.key_dim))
        value_val = np.reshape(value_val, (seq_len, self.num_heads, self.key_dim))

        # transpose so that (num_heads, seq_len, d_model)
        query_val = np.transpose(query_val, axes=(1, 0, 2))
        key_val = np.transpose(key_val, axes=(1, 0, 2))
        value_val = np.transpose(value_val, axes=(1, 0, 2))

        # # pre-matmul scaling
        # query_val = query_val / np.sqrt(self.key_dim)

        # dot product
        attention_score = MatMul.forward(query_val, key_val.transpose(0, -1, -2))
        # post-matmul scaling
        attention_score = attention_score / np.sqrt(self.key_dim)
        # masking
        attention_mask = ~attention_mask.astype(bool)
        attention_score[:, :, attention_mask] = np.finfo(attention_score.dtype).min   
        # softmax
        attention_score = Softmax.forward(attention_score)

        # dot product between attention scores and values
        weighted_val = MatMul.forward(attention_score, value_val)

        # concatenation along embeddings
        concatenated_val = np.reshape(np.transpose(weighted_val, axes=(1, 0, 2)), (seq_len, self.dimension))

        # output linear transformation
        linear = self.linear.forward(concatenated_val)

        return linear

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        dY = self.linear.backward(dY)

        # Backward pass through each attention head
        dY_splits = np.array(np.split(dY, self.num_heads, axis=0))
        
        gradients = [head.backward(dY_split, Y, Y_hat, loss_function) for head, dY_split in zip(self.attention_heads, dY_splits)]

        # Sum the gradients from each attention head
        total_gradient = np.sum(gradients, axis=0)

        return total_gradient
    
    """
        This layer has a total of num_heads * 6 + 2 trainable variables, 6 trainable variables per head, 2 trainable variables for the dense linear.

        :arg qkv_weights_bias [should be trainable variables of the whole multi-head attention layer in a list.]
    """

    def get_trainable_variables(self):
        trainable_variables = {}

        query_trainable_variables = self.query.get_trainable_variables()
        trainable_variables['multi_head/query/dense/weights'] = query_trainable_variables['dense/weights']
        trainable_variables['multi_head/query/dense/bias'] = query_trainable_variables['dense/bias']

        key_trainable_variables = self.key.get_trainable_variables()
        trainable_variables['multi_head/key/dense/weights'] = key_trainable_variables['dense/weights']
        trainable_variables['multi_head/key/dense/bias'] = key_trainable_variables['dense/bias']

        value_trainable_variables = self.value.get_trainable_variables()
        trainable_variables['multi_head/value/dense/weights'] = value_trainable_variables['dense/weights']
        trainable_variables['multi_head/value/dense/bias'] = value_trainable_variables['dense/bias']

        output_trainable_variables = self.linear.get_trainable_variables()
        trainable_variables['multi_head/output/dense/weights'] = output_trainable_variables['dense/weights']
        trainable_variables['multi_head/output/dense/bias'] = output_trainable_variables['dense/bias']

        return trainable_variables
    
    def set_trainable_variables(self, qkv_weights_bias: List[NDArray[np.float32]]):
        # revise this dimension checking
        # if (self.num_heads * 6 + 2 != len(qkv_weights_bias)):
        #     raise ValueError(f"Expected " + str(self.num_heads * 6 + 2) + " variables, received " + str(len(qkv_weights_bias)))
        
        # if any(type(variable) is not np.ndarray for variable in qkv_weights_bias):
        #     raise ValueError(f"A none np.ndarray type variable is found in the list.")

        self.query.set_trainable_variables(qkv_weights_bias[0], qkv_weights_bias[1])
        self.key.set_trainable_variables(qkv_weights_bias[2], qkv_weights_bias[3])
        self.value.set_trainable_variables(qkv_weights_bias[4], qkv_weights_bias[5])
            
        self.linear.set_trainable_variables(qkv_weights_bias[6], qkv_weights_bias[7])

    def reinstantiate(self):
        return type(self)(self.num_heads, self.sequence_len, self.dimension)


"""
   Layer normalization is a technique used to normalize the activations or 
   outputs of each layer in a neural network thus stabilizing the training process, 
   improving convergence, and making the model less sensitive to the scale of the inputs.
   For further information, please refer to https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/
   
   Note: This layer is still on experimental phase.
"""

# this is the problem!
class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.G = np.ones((1,))  # Initialized to 1
        self.b = np.zeros((1,))  # Initialized to 0

    def set_optimizer(self, optimizer):
        pass

    def forward(self, X, prev_X, training=True):
        if prev_X is not None:
            X = (X + prev_X)
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        self.normalized_input = np.divide((X - mean), np.sqrt(variance + self.epsilon))
        output = self.G * self.normalized_input + self.b
        return output

    def backward(self, dY):
        dY = dY.T
        dX_normalized = dY * self.G
        dvariance = np.sum(dX_normalized * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)), axis=-1, keepdims=True) * -0.5 * np.power(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon, -1.5)
        dmean = np.sum(dX_normalized * -1 / np.sqrt(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon), axis=-1, keepdims=True) + dvariance * np.sum(-2 * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)), axis=-1, keepdims=True) / self.normalized_input.shape[-1]
        dX = dX_normalized / np.sqrt(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon) + dvariance * 2 * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)) / self.normalized_input.shape[-1] + dmean / self.normalized_input.shape[-1]
        
        # Gradient update for scale and shift would typically be here

        return dX.T
    
    def get_trainable_variables(self):
        return {
            'layer_normalization/gamma': self.G, 
            'layer_normalization/beta': self.b
        }

    def set_trainable_variables(self, gamma: NDArray[np.float32], beta: NDArray[np.float32]):
        self.G = gamma
        self.b = beta

    def reinstantiate(self):
        return type(self)