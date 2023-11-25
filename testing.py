import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

inputs = torch.randn(4, 1, 2, 3, 2)

m = nn.Flatten()
output = m(inputs)

output.size()

print(inputs)
print(output)