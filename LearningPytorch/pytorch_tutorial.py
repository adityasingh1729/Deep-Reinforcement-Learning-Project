import torch
import numpy as np

# x = torch.zeros(2,3)
# y = torch.ones(2,4,5)
# z = torch.ones(3,4, dtype=torch.int)
# w = torch.empty(4,3,2,3)

# print(x)
# print(y.dtype)
# print(z.dtype)
# print(w)

# print(x.size())

# v = torch.tensor([2.3,4.5,8])
# print(v)

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x)
# print(y)
# z = x + y
# w = torch.add(x,y)
# z = x - y
# w = torch.sub(x,y)
# torch.mul(x,y)
# y.mul_(x)
# torch,div(x,y)
# print(z)
# print(w)

# y.add_(x)   # this is y += x, functions followed by underscore perform inplace operations
# print(x)
# print(y)

### Tensor Slicing
# x = torch.rand(5,3)
# print(x)
# print(x[2:4,1:3])
# if a tensor has only one element, then we can use .item() to get that element, whereas x[2,3] gives 
# one element but that is a tensor

### Resizing tensors:
# x = torch.rand(4,4)
# y = x.view(2,8)
# y = x.view(-1,8) # here it itself determines the dimension where we put -1

### Tensor to numpy array and vice versa
# a = torch.ones(5)
# b = a.numpy() # The important fact to remember is that a and b share the same location in memory
# so when one is modified the other also gets modified
# vice versa:
# a = np.ones(5)
# b = torch.from_numpy(a)  or b = torch.from_numpy(a, dtype=torch.int)

### Another command which is required later is:
# x = torch.ones(5, requires_grad = True) # This is required when we want to optimize, by calculating the gradients