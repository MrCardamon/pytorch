import numpy
import torch

x= torch.empty(2,2)
print(x) 

x= torch.ones(2,2, dtype = torch.float16)
print(x.dtype)

x= torch.rand(2,2)
y= torch.rand(2,2)

z= torch.add(x,y)
print(z)

y.add_(x) # any _ in torch is inplace
print(y)
