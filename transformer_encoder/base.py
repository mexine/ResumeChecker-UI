import json
import numpy as np
from numpy.typing import NDArray
from typing import List, Callable, Tuple
import time

# project imports
from transformer_encoder.loss import CategoricalCrossEntropy
from transformer_encoder.optimizer import GradientDescent
from transformer_encoder.layers import Layer, Dense, WordEmbedding, PositionalEncoding, MultiHeadAttention, SelfAttention, LayerNormalization
from transformer_encoder.activation import Softmax


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def fit(self,
            X: NDArray[np.float64],
            attention_mask: NDArray[np.float64],
            mlm_mask: NDArray[np.float64],
            Y: NDArray[np.float64],
            loss_function = CategoricalCrossEntropy,
            optimizer = GradientDescent(),
            accuracy_metric: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = None) -> str:
        """
            Accepts a single sequence. Returns training log.
        """
        start_time = time.time() # runtime log

        layer_count = len(self.layers)  # get number of layers
        # transpose so that matrix size is (num of input, num of samples)
        X = X.T
        Y = Y.T

        for layer in self.layers:
            layer.set_optimizer(optimizer=optimizer)  # set each layer optimizers

        Y_hat = X
        prev_Y_hat = None
        
        # forward propagation
        for index, layer in enumerate(self.layers):
            # check index to avoid index error
            if (index + 2 < layer_count):
                # store Y_hat before FFN
                if (type(self.layers[index + 1]) is Dense and
                    type(self.layers[index + 2]) is LayerNormalization):
                    prev_Y_hat = Y_hat

            if (type(layer) == MultiHeadAttention or
                type(layer) == SelfAttention):
                prev_Y_hat = Y_hat # store Y_hat before Attention layer
                Y_hat = layer.forward(Y_hat, attention_mask=attention_mask)
            elif type(layer) == LayerNormalization:
                Y_hat = layer.forward(Y_hat, prev_Y_hat)
            else:
                Y_hat = layer.forward(Y_hat)

        # apply mlm_mask
        mlm_masked_Y_hat = Y_hat * mlm_mask.T
        mlm_masked_Y_hat[mlm_masked_Y_hat == 0] = 10**-100

        # backward propagation
        dY = self.layers[layer_count - 1].backward(np.array([]), Y, mlm_masked_Y_hat, loss_function)
        for j in range((layer_count - 2), -1, -1):
            dY = self.layers[j].backward(dY)

        # logger
        accuracy_log = ''
        if accuracy_metric is not None:
            accuracy_log = ', Accuracy: ' + str(accuracy_metric(Y, Y_hat))  # report accuracy

        return 'Loss: ' + str(loss_function.forward(Y, mlm_masked_Y_hat)) + accuracy_log + ', Elapsed Time: ' + str(time.time() - start_time)

    def predict(self, 
                X: List[NDArray[np.float64]],
                attention_masks: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        """
            Accepts list of truncated resume. Expects tokenizer to handle truncation.
        """

        layer_count = len(self.layers)  # get number of layers

        outputs = []
        for x, mask in zip(X, attention_masks):
            x = x.T # transpose so that size is (num of input, num of samples)
            prev_x = None

            # forward propagation
            for index, layer in enumerate(self.layers):
                # check index to avoid index error
                if (index + 2 < layer_count):
                    # store Y_hat before FFN
                    if (type(self.layers[index + 1]) is Dense and
                        type(self.layers[index + 2]) is LayerNormalization):
                        prev_x = x

                if (type(layer) == MultiHeadAttention or
                    type(layer) == SelfAttention):
                    prev_x = x # store Y_hat before Attention layer
                    x = layer.forward(x, attention_mask=mask, training=False)
                elif type(layer) == LayerNormalization:
                    x = layer.forward(x, prev_X=prev_x, training=False)
                else:
                    x = layer.forward(x, training=False)
            
            outputs.append(x)

        return np.mean(outputs, axis=0)

    def validate(self,
                 X: NDArray[np.float64],
                 attention_mask: NDArray[np.float64],
                 mlm_mask: NDArray[np.float64],
                 Y: NDArray[np.float64],
                 loss_function = CategoricalCrossEntropy,
                 accuracy_metric: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = None):
        V, V_hat = Y.T, X.T
        prev_V_hat = None
        layer_count = len(self.layers)  # get number of layers
        
        # forward propagation
        for index, layer in enumerate(self.layers):
            # check index to avoid index error
            if (index + 2 < layer_count):
                # store V_hat before FFN
                if (type(self.layers[index + 1]) is Dense and
                    type(self.layers[index + 2]) is LayerNormalization):
                    prev_V_hat = V_hat

            if (type(layer) == MultiHeadAttention or
                type(layer) == SelfAttention):
                prev_V_hat = V_hat # store V_hat before Attention layer
                V_hat = layer.forward(V_hat, attention_mask=attention_mask, training=False)
            elif type(layer) == LayerNormalization:
                V_hat = layer.forward(V_hat, prev_V_hat, training=False)
            else:
                V_hat = layer.forward(V_hat, training=False)

        # apply mlm_mask
        mlm_masked_V_hat = V_hat * mlm_mask.T
        mlm_masked_V_hat[mlm_masked_V_hat == 0] = 10**-100

        return loss_function.forward(V, mlm_masked_V_hat), accuracy_metric(V, V_hat)

    def remove_mlm_head(self) -> None:
        """
            Removes MLM head. Can't be used when model doesn't have Dense with Softmax as last layer.
        """

        if (type(self.layers[-1]) is Dense 
            and self.layers[-1].activation is Softmax):
            self.layers = self.layers[:-1]
        else:
            raise ValueError("The model doesn't have an MLM Head.")
    
    def get_trainable_variables(self):
        trainable_variables = {}
        for index, layer in enumerate(self.layers):
            layer_trainable_variables = layer.get_trainable_variables()
            if (layer_trainable_variables != None):
                for trainable_variable in layer_trainable_variables:
                    trainable_variables[str(index) + '/' + trainable_variable] = layer_trainable_variables[trainable_variable]

        return trainable_variables

    def save_model(self, file_name: str):
        with open('model_weights/' + file_name, 'w') as file:
            trainable_variables = self.get_trainable_variables()

            for key in trainable_variables:
                file.write(str(trainable_variables[key].tolist()) + '\n')

    def load_model(self, file_name: str):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            index = 0
            for layer in self.layers:
                if type(layer) == Dense:
                    weights = np.array(json.loads(lines[index]), dtype=np.float64)
                    bias = np.array(json.loads(lines[index + 1]), dtype=np.float64)
                    layer.set_trainable_variables(weights, bias)
                    index += 2
                elif type(layer) == WordEmbedding:
                    weights = np.array(json.loads(lines[index]), dtype=np.float64)
                    layer.set_trainable_variables(weights)
                    index += 1
                elif type(layer) == PositionalEncoding:
                    pass
                elif type(layer) == MultiHeadAttention:
                    num_trainable_variables = layer.num_heads * 6 + 2
                    qkv_weights_bias = lines[index:index + num_trainable_variables]
                    for qkv_index, trainable_variable in enumerate(qkv_weights_bias):
                        qkv_weights_bias[qkv_index] = np.array(json.loads(trainable_variable), dtype=np.float64)
                    layer.set_trainable_variables(qkv_weights_bias)
                    index += num_trainable_variables
                elif type(layer) == LayerNormalization:
                    pass
                else:
                    raise ValueError('Layer not implemented in load_model().')