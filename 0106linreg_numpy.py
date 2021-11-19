import torch

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_numpy, y_numpy= datasets.make_regression(n_samples=100, n_features= 1, noise = 20, random_state= 1)

X= torch.from_numpy(x_numpy.astype(np.float32))
y= torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features= X.shape
input_size= n_features
output_size= 1
model = torch.nn.Linear(input_size, output_size)

criterion = torch.nn.MSELoss()
learning_rate = 0.01
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate )

for i in range(1000):
    y_hat= model(X)
    loss= criterion(y, y_hat)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if i%10 ==0:
        print(f'iteration {i} loss: {loss.item():0.4f}')

    
predicted = model(X).detach() # gradient attribute false
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted,'b')


