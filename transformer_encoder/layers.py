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
                X: NDArray[np.float64],
                training: bool = True) -> NDArray[np.float64]:
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
                 dY: NDArray[np.float64],
                 Y: NDArray[np.float64] = None,
                 Y_hat: NDArray[np.float64] = None,
                 loss_function: Loss = None) -> NDArray[np.float64]:
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
    
    def set_trainable_variables(self, weights: NDArray[np.float64], bias: NDArray[np.float64]):
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
        self.W = np.random.uniform(-1, 1, (embedding_dim, vocab_size))
        self.X = None
        self.optimizer = GradientDescent()

        self.trainable_variables = {
            'word_embedding/weights': self.W
        }

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer.get_optimizer(self.W.shape)

    def forward(self, X: NDArray[np.int64], training=True):
        self.X = X
        return self.W[:, X]

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

    def set_trainable_variables(self, weights: NDArray[np.float64]):
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
                 eta: int = 10000,
                 bptt: bool = False):
        temp = np.zeros((sequence_len, dimension), dtype=np.float64)  # initialize positional encodings
        for k in range(sequence_len):
            for i in np.arange(int(dimension / 2)):
                denominator = np.power(eta, 2 * i / dimension)
                temp[k, 2 * i] = np.sin(k / denominator)
                temp[k, 2 * i + 1] = np.cos(k / denominator)

        self.dimension = dimension
        self.pos_encoding = temp.T  # cache positional encodings

    def set_optimizer(self, optimizer):
        pass

    def get_positional_encoding(self):
        return self.pos_encoding

    def forward(self, X, training=True):
        return X + self.pos_encoding

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        return dY
    
    def get_trainable_variables(self):
        pass

    def set_trainable_variables(self):
        pass
    
    def reinstantiate(self):
        raise ValueError('PositionalEncoding layers are typically not stacked.')


"""
    Self Attention Layer is a fundamental layer of transformer-based architectures
    that enables the model to weigh the importance of different words or tokens in a sequence based on their 
    relationships with each other. For additional details please refer to https://research.google/pubs/pub46201/
    
    Note: This layer is still on experimental phase.
"""


class SelfAttention(Layer):
    def __init__(self,
                 sequence_len: int,
                 dimension: int):
        self.query = Dense((dimension, dimension), Linear())
        self.key = Dense((dimension, dimension), Linear())
        self.value = Dense((dimension, dimension), Linear())

        # cache values to compute derivatives
        self.query_val = []
        self.key_val = []
        self.value_val = []

        self.sequence_len = sequence_len
        self.dimension = dimension
        # self.attention_scores = []  # cache scores
        self.attention_scores = None  # caching of previous scores

    def set_optimizer(self, optimizer):
        pass

    def get_attention_scores(self):
        return np.array(self.attention_scores)

    @staticmethod
    def __attention_derivative(attention: NDArray[np.float64], dY: NDArray[np.float64]):
        dY_hat = []
        for i in range(len(attention)):
            dS = np.reshape(attention[i], (-1, 1))
            dS = np.diagflat(dS) - np.dot(dS, dS.T)
            dY_hat.append(dY[i].dot(dS))

        return np.array(dY_hat)

    def forward(self, X, attention_mask = None, training=True):
        flatten_seq = np.reshape(X, (-1, self.dimension))
        self.query_val = self.query.forward(flatten_seq.T).T
        self.key_val = self.key.forward(flatten_seq.T).T
        self.value_val = self.value.forward(flatten_seq.T).T
    
        # MatMul of Q & K
        attention_score = MatMul.forward(self.query_val, self.key_val.T)

        # Scaling of values
        attention_score = (1 / self.dimension ** 0.5) * attention_score

        if (attention_mask is not None):
            attention_score *= attention_mask

        self.attention_score = Softmax.forward(attention_score.T).T

        weighted_value = MatMul.forward(self.attention_score, self.value_val).T

        return np.array(weighted_value)

    def backward(self, dY, Y=None, Y_hat=None, loss_function=None):
        dA, dV = MatMul.backward(dY.T, self.attention_score, self.value_val)
        dA = (1 / self.dimension ** 0.5) * self.__attention_derivative(self.attention_score, dA)
        dQ, dK = MatMul.backward(dA, self.query_val, self.key_val.T)

        return self.query.backward(dQ.T) + self.key.backward(dK) + self.value.backward(dV.T)
    
    """
        This layer has a total of 6 trainable variables, 2 trainable variables per Q, K, V, Weights and bias.
    """

    def get_trainable_variables(self):
        self_attention_query_weights_bias = self.query.get_trainable_variables()
        self_attention_key_weights_bias = self.key.get_trainable_variables()
        self_attention_value_weights_bias = self.value.get_trainable_variables()
        return {
            'self_attention/query/dense/weights': self_attention_query_weights_bias['dense/weights'],
            'self_attention/query/dense/bias': self_attention_query_weights_bias['dense/bias'],
            'self_attention/key/dense/weights': self_attention_key_weights_bias['dense/weights'],
            'self_attention/key/dense/bias': self_attention_key_weights_bias['dense/bias'],
            'self_attention/value/dense/weights': self_attention_value_weights_bias['dense/weights'],
            'self_attention/value/dense/bias': self_attention_value_weights_bias['dense/bias'],
        }

    def set_trainable_variables(self, query_weights: NDArray[np.float64], query_bias: NDArray[np.float64],
                                key_weights: NDArray[np.float64], key_bias: NDArray[np.float64],
                                value_weights: NDArray[np.float64], value_bias: NDArray[np.float64]):
        # add dimension checking here...

        self.query.set_trainable_variables(query_weights, query_bias)
        self.key.set_trainable_variables(key_weights, key_bias)
        self.value.set_trainable_variables(value_weights, value_bias)

    def reinstantiate(self):
        return type(self)(self.sequence_len, self.dimension, self.masked, self.bptt)
     

"""
    Multi-Head Attention is a parrallelized version of the Self Attention Layer. 
    It has num_heads number of heads or parrallelized Self Attention Layer. 

    Note: This layer is still on experimental phase.
"""

class MultiHeadAttention(Layer):
    def __init__(self, num_heads: int, sequence_len: int, dimension: int):
        self.num_heads = num_heads
        self.sequence_len = sequence_len
        self.dimension = dimension
        self.attention_heads = []

        for _ in range(num_heads):
            self.attention_heads.append(SelfAttention(sequence_len=sequence_len, dimension=dimension))

        self.linear = Dense((num_heads * dimension, dimension), Linear())

    def set_optimizer(self, optimizer):
        pass

    def forward(self, X, attention_mask=None, training=True):
        # Forward pass through each attention head
        head_outputs = [head.forward(X, attention_mask=attention_mask, training=training) for head in self.attention_heads]

        # Concatenate the outputs along the last axis
        concatenated_output = np.concatenate(head_outputs, axis=0)

        transformed_output = self.linear.forward(concatenated_output)

        return transformed_output

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

        for index, head in enumerate(self.attention_heads):
            head_trainable_variables = head.get_trainable_variables()
            for trainable_variable in head_trainable_variables:
                trainable_variables['multi_head_' + str(index) + '/' + trainable_variable] = head_trainable_variables[trainable_variable]

        linear_trainable_variables = self.linear.get_trainable_variables()
        for trainable_variable in linear_trainable_variables:
            trainable_variables['multi_head/' + trainable_variable] = linear_trainable_variables[trainable_variable]

        return trainable_variables
    
    def set_trainable_variables(self, qkv_weights_bias: List[NDArray[np.float64]]):
        # add dimension checking here...
        if (self.num_heads * 6 + 2 != len(qkv_weights_bias)):
            raise ValueError(f"Expected " + str(self.num_heads * 6 + 2) + " variables, received " + str(len(qkv_weights_bias)))
        
        if any(type(variable) is not np.ndarray for variable in qkv_weights_bias):
            raise ValueError(f"A none np.ndarray type variable is found in the list.")

        for index, head in enumerate(self.attention_heads):
            head.set_trainable_variables(
                qkv_weights_bias[index * 6], 
                qkv_weights_bias[index * 6 + 1],
                qkv_weights_bias[index * 6 + 2],
                qkv_weights_bias[index * 6 + 3],
                qkv_weights_bias[index * 6 + 4],
                qkv_weights_bias[index * 6 + 5])
            
        self.linear.set_trainable_variables(qkv_weights_bias[-2], qkv_weights_bias[-1])

    def reinstantiate(self):
        return type(self)(self.num_heads, self.sequence_len, self.dimension)


"""
   Layer normalization is a technique used to normalize the activations or 
   outputs of each layer in a neural network thus stabilizing the training process, 
   improving convergence, and making the model less sensitive to the scale of the inputs.
   For further information, please refer to https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/
   
   Note: This layer is still on experimental phase.
"""


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.G = np.ones((1,))  # Initialized to 1
        self.b = np.zeros((1,))  # Initialized to 0

    def set_optimizer(self, optimizer):
        pass

    def forward(self, X, prev_X, training=True):
        X = (X + prev_X).T
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        self.normalized_input = (X - mean) / np.sqrt(variance + self.epsilon)
        output = self.G * self.normalized_input + self.b
        return output.T

    def backward(self, dY):
        dY = dY.T
        dX_normalized = dY * self.G
        dvariance = np.sum(dX_normalized * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)), axis=-1, keepdims=True) * -0.5 * np.power(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon, -1.5)
        dmean = np.sum(dX_normalized * -1 / np.sqrt(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon), axis=-1, keepdims=True) + dvariance * np.sum(-2 * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)), axis=-1, keepdims=True) / self.normalized_input.shape[-1]
        dX = dX_normalized / np.sqrt(np.var(self.normalized_input, axis=-1, keepdims=True) + self.epsilon) + dvariance * 2 * (self.normalized_input - np.mean(self.normalized_input, axis=-1, keepdims=True)) / self.normalized_input.shape[-1] + dmean / self.normalized_input.shape[-1]
        
        # Gradient update for scale and shift would typically be here

        return dX.T
    
    def get_trainable_variables(self):
        pass

    def set_trainable_variables(self):
        pass

    def reinstantiate(self):
        return type(self)