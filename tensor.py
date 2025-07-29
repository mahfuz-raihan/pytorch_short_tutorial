# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 00:54:29 2025

@author: Mahfuz
"""

import torch
print(f"torch version: {torch.__version__}")

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, Using CPU")

# Create tensor

# using empty
# --------------------------------------------
a = torch.empty(2,3)
print(f"Tensor type: {type(a)}")
# using zeros tensor
print(f"Zeros matrix ====> \n{torch.zeros(2,3)}")

# using ones
# --------------------------------------------
print(f"One matrix ====> \n{torch.ones(2,3)}")

# using random tensor
# --------------------------------------------
print(f"Random matrix ====> \n{torch.rand(2,3)}")

# Using of seed
# --------------------------------------------
print(f"Random matrix ====> \n{torch.rand(2,3)}")

torch.manual_seed(100)
print(f"Seed Random matrix ====> \n{torch.rand(2,3)}")

torch.manual_seed(100)
print(f"Seed Random matrix ====> \n{torch.rand(2,3)}")

# Creating tensor
# --------------------------------------------
print(f"Tensor ====> \n{torch.tensor([[1,2,3],[4,5,6]])}")

# other ways
# --------------------------------------------
# arange -------------------------------------
print(f"Create tensor using arange ====> \n{torch.arange(0,11,2)}")

# linspace -----------------------------------
print("Create tensor using linspace ====> \n", torch.linspace(0,10,10))


# eye ----------------------------------------
print("using eye ====> \n", torch.eye(5))

# using full ---------------------------------
print("Using full ====> \n", torch.full((3,3),5))


# Tensor shape
# --------------------------------------------
x = torch.tensor([[1,2,3],[4,5,6]])
print("The shape of the tensor x: \n", x.shape)


# Tensor Data Type
# --------------------------------------------
print("Find data type:", x.dtype)

# assign data type
# --------------------------------------------
print("Assigned Data Type:", torch.tensor([1.0, 2.0, 3.0], dtype=torch.int32))
