# Here we solve the Binary Classification problem.
# Binary classification is like - is it a dog or not.
# Here we use the BCELoss and we need to take the sigmoid at the end.
# If output is < 0.5 not a dog and otherwise it is a dog.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary Classification

## Method 1:
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
# Like above we used nn.ReLU() and nn.Sigmoid, we also have:
# nn.Softmax(), nn.TanH and nn.LeakyReLU

## Method2: Using in-built pytorch functions for the activation functions
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
    
# Like above we used torch.relu() and torch.sigmoid(), we also have:
# torch.softmax(), torch.tanh()
# For leaky ReLU it is not in torch, instead in torch.nn.functional, 
# so we need to do: torch.nn.functional.leaky_relu()
# instead we may import torch.nn.functional as F and then we have:
# F.relu(), F.sigmoid(), F.tanh(), F.softmax(), F.leaky_relu()
    
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()