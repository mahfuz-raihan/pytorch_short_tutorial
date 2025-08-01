# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 02:04:43 2025

@author: Mahfuz
"""

# What is autograd in pyTorch? 
"""
autograd is a core component of PyTorch that provides automatic differentiation 
for tensor operations, it enables gradient computation, which is essential for
training machine learning models using optimization algorithms like gradient descent.
"""

import torch
x = torch.tensor(3.0, requires_grad=True)
y = x**2

print("x -", x)
print("y -", y)
