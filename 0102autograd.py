import torch

x= torch.randn(2, requires_grad=True)
print(x)


y= x+ 2

z= y*y*2
print(z)

v= torch.tensor([0.1,1.0], dtype = torch.float32)
z.backward(v) #dz/dx
print(x.grad)


# sstops pytorch from tracking and creating history 
with torch.no_grad():
    y= x + 2
    print(y)


weights = torch.ones(4, requires_grad= True)

for epoch in range(1):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.zero_grad() # otherwise gradients accumulate