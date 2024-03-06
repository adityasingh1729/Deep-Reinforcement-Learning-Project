import torch
import torch.nn as nn
import numpy as np

# Here we study cross-entropy loss which is associated with the softmax function.
# Cross-Entropy Loss = -1/N * SIGMA(Y_i * log(Y_i))
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # here we ignore the 1/N, that is we don't normalise. if we wanted to we could divide by predicted.shape[0]

# for cross-entropy loss, y must be one hot encoded, which means that exactly one of the classes must be 1 and the others 0
# if class 1: [1,0,0] 
# if class 2: [0,1,0] 
# if class 3: [0,0,1]
Y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])  # these are the values obtained by softmax
Y_pred_bad = np.array([0.1, 0.3, 0.6])  # these are the values obtained by softmax
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

# Here we are solving a classification problem
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')