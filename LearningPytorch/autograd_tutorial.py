### Here we will learn how to use the autograd to calculate the gradients of the tensors which are helpful in optimizations

import torch

x = torch.randn(3, requires_grad=True)
print(x)

# ### Now whenever we use x in a computation pytorch creates a computational graph for us
# y = x + 2
# z = y*y*2
# # w = z.mean()
# print(y)
# print(z)
# print(w)

# # w.backward()  # dz/dx because w is mean of z
# # but what if we didn't want to create mean of z, then we multiply it with a Jacobian to convert it to
# # a 1 X 1 matrix, basically we can take gradient of a scalar or a 1 x 1 matrix
# v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
# z.backward(v)
# print(x.grad)

# What if later we don't want grad for x, 3 ways:
# 1: x.requires_grad_(False)
# 2: y = x.detach() -- creates a new tensor y having same value as x and not requiring grad
# 3: After this method x still requires grad, but it creates a space where the other variables which deal with x don't require grad
# with torch.no_grad():
#     y = x + 2   # y doesn't require grad
#     print(y)
# print(x)    # x still requires grad

# if we call z.backward() again and again then the result of each call is accumulated in x.grad, giving the 
# wrong result. So we need to call x.grad.zero_() to empty x.grad every time