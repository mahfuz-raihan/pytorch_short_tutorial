import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)
# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

