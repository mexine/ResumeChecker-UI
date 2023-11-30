import numpy as np
import tensorflow as tf

from loss import CategoricalCrossEntropy
from activation import Softmax

loss = CategoricalCrossEntropy()
activation = Softmax()

y_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]).T
y_pred = np.array([[0.04, 0.95, 0.01], [0.1, 0.8, 0.1], [10**-8, 10**-8, 10**-8], [10**-8, 10**-8, 10**-8]]).T

# y_pred = activation.forward(y_pred.T).T

# print(np.exp(0.04))
# print(np.exp([[0.04, 0.95, 0.01], [0.1, 0.8, 0.1], [10**-8, 10**-8, 10**-8]]))
# print(np.exp(np.array([[0.04, 0.95, 0.01], [0.1, 0.8, 0.1], [10**-100, 10**-100, 10**-100]])))
# print(np.exp(y_pred))
# print(y_pred)

# y_true = np.zeros((50, 32000), dtype=np.float64)
# for index in range(4):
#     y_true[index][np.random.randint(0, 49)] = 1
# a = np.random.rand(1, 32000)
# a = activation.forward(a.T).T
# b = np.random.rand(1, 32000)
# b = activation.forward(b.T).T
# y_pred = np.append(a, np.full((1, 32000), 10**-100), axis=0)
# y_pred = np.append(y_pred, b, axis=0)
# y_pred = np.append(y_pred, np.full((47, 32000), 10**-100), axis=0)

# y_true = np.zeros((4, 32000), dtype=np.float64)
# for index in range(4):
#     y_true[index][np.random.randint(0, 49)] = 1
# y_pred = np.random.rand(4, 32000)
# y_pred = activation.forward(y_pred.T).T

# y_true_add = np.zeros((46, 32000), dtype=np.float64)
# y_true2 = np.append(y_true, y_true_add, axis=0)

# y_pred_add = np.full((46, 32000), 10**-100)
# y_pred2 = np.append(y_pred, y_pred_add, axis=0)

# print(y_true)
# print(y_true.shape)

# print(y_pred)
# print(y_pred.shape)

# for row in y_true:
#     print(row)

# for row in y_pred:
#     print(row)

print(tf.keras.losses.CategoricalCrossentropy()(y_true.T, y_pred.T).numpy())
print(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true.T, y_pred.T).numpy())
print(np.mean(-np.sum(y_true.T * np.log(y_pred.T), axis=-1)))
print(loss.forward(y_true.T, y_pred.T))



# MEAN Expected Loss: 0.588469596845399
# SUM Expected Loss: 2.353878387381596