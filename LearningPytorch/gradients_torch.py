### Training pipeline in pytorch:
# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn

# Here we take f = 2 * x

# X = torch.tensor([1,2,3,4], dtype=torch.float32)
# Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# The below 4 lines are now not needed because we now use a pytorch model which contains the model predictor and the weights
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# # model prediction
# def forward(x):
#    return w*x

# We require X and Y in new shape, here outer [] contains the various samples
# and the inner [] contain the features in each sample, so here we have 4 samples with 1 feature each
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# Here we have only 1 layer, so we don't have t make our own model, 
# we can use the builtin model in pytorch: nn.Linear()
# model = nn.Linear(input_size, output_size)

# or like below we can create our own model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__() # this is a super constructor
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 200

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights, with statement is required because updating w shouldn't be a part of backward calculation
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    # The above is the manual method to update weights, now we can use the optimizer
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    if epoch % 20 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')