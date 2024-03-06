import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()
# while using the above function we must remember some important things
# this functions applies the softmax function by itself, so we don't need to do that manually
# Y must not be one hot encoded, instead we give it the class member which is requried.

# 3 samples
Y = torch.tensor([2,0,1])  # not one hot encoded instead the class, here 0, is given

# for Y_pred we give a tensor with dim = n_samples x n_classes, here 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]]) # good because the class 0 has the max value
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]]) # bad because the class 0 is not the one with the max value

# gives the cross-entropy loss
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad ,Y)

print(l1.item())
print(l2.item())

# gives the prediction for the classes
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)