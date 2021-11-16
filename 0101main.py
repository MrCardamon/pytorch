import numpy as np
import torch

"""
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
y = x.view(4)
print(y)
print(y.size())
"""



a= torch.ones(5)
b = a.numpy()
a.add_(1)

print(b)


if torch.cuda.is_available():
    device= torch.device("cuda")
    y= torch.ones(5, device = device)
    # or 
    x = torch.ones(5)
    x= x.to(device)

    z= x+y
    z= z.to("cpu")
    z=z.numpy()


x=torch.ones(5, requires_grad= True)
# tells pytorch that need to calculate gradient
# for variables you need to optimize


