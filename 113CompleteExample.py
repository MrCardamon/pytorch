#%%
import torch
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader  
import numpy as np
import matplotlib.pyplot as plt
#%%
class RegressionDatatset(Dataset):
    def __init__(self,n_samples=100, n_features= 1, noise = 20, random_state= 1):
        x_numpy, y_numpy= datasets.make_regression(n_samples=n_samples, n_features= n_features, noise = noise, random_state= random_state)

        self.X= torch.from_numpy(x_numpy.astype(np.float32))
        self.y= torch.from_numpy(y_numpy.astype(np.float32))
        self.y = self.y.view(-1,1)
        
        self.n_samples= self.X.shape[0]

    def __getitem__(self, index): #first_data= mydataset[0]
        return self.X[index], self.y[index]
    def __len__(self):
        return self.n_samples


#%%
n_samples=120
n_features= 3
batch_size= 4
num_epochs= 10
mydataset = RegressionDatatset(n_samples=n_samples, n_features= n_features)
dataloader= DataLoader(dataset= mydataset, batch_size= batch_size, shuffle= True)
model= torch.nn.Linear(in_features= n_features, out_features= 1)
lr=0.01
optimizer= torch.optim.SGD(params= model.parameters(), lr=lr)
criterion= torch.nn.MSELoss()
# %%
n_iterations= len(dataloader)
assert(np.array_equal(n_iterations, len(mydataset)/batch_size))
for epoch in range(num_epochs):
    for i, (inputs, y) in enumerate(dataloader):
        #forward, backward, update
        
        model.zero_grad()
        yhat= model(inputs)
        loss= criterion(yhat, y)
        loss.backward()
        optimizer.step()
        if (i+1)%5== 0: 
            print(f"epoch {epoch}/{num_epochs} , step {i+1}/{n_iterations}, inputs {inputs.shape}")

# %%
with torch.no_grad():
    yhat= model(mydataset.X)
plt.scatter(mydataset.y, yhat)

plt.figure()
plt.hist((mydataset.y -yhat).view(-1), bins= 100)
plt.show()