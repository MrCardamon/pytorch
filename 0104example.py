import torch

x= torch.tensor([1,2,3], dtype= torch.float32)
ws= torch.tensor(2)
y = (x*ws).mean()
print(y)

def forward(x,w):
    return (x*w).mean()

def loss(y,y_hat):
    return ((y-y_hat)**2).mean()

w= torch.tensor(0.0, requires_grad= True, dtype= torch.float32) # leaf variable
learning_rate= torch.tensor(0.05, dtype=torch.float32)
for epoch in range(40):
    y_hat = forward(x,w)
    l=loss(y,y_hat)

    l.backward()
    with torch.no_grad(): 
# PyTorch doesn’t allow in-place operations on leaf variables that have requires_grad=True 
        w+= -learning_rate * w.grad

    w.grad.zero_()
    print(f' iteration: {epoch} loss: {l} w: {w}')