# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 00:51:16 2025

@author: mahfuz
"""
# %%
import torch
print(torch.__version__)
x = torch.rand(5, 3)
print(x)


# %%
import torch
# Create a 2D tensor (matrix)
scalar = torch.tensor(7)
print(scalar)
# %%
print(scalar.item())  # Convert to Python number
# %%
# 2d vector
vector = torch.tensor([1, 2, 3])
print(vector)
# %%
# 2d matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)
# %%
# 3d tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4 ]], [[5, 6], [7, 8]]])
print(tensor_3d)
print(tensor_3d.ndim)  # Number of dimensions
# %%
