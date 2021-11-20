import torch
import torch.nn as nn




X = torch.tensor([[1],[2],[3]], dtype= torch.float32)
Ws= torch.tensor(2.0)
Y = X*Ws
print(Y)

w= torch.tensor(0, requires_grad= True, dtype= torch.float32)

n_samples, n_features= X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

X_test= torch.tensor([5], dtype= torch.float32)


input_size= n_features
output_size= n_features

model= nn.Linear(input_size, output_size)

print(f'Prediction before training: f({X_test[0]}) = {model(X_test).item():.3f}')
#print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate= 0.001
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
loss = nn.MSELoss()
for epoch in range(100):
    y_hat= model(X)

    l= loss(Y, y_hat)

    l.backward()

    optimizer.step()

    if epoch %10== 0:
        [w,b]= model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')