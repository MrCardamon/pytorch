# https://pytorch.org/docs/stable/data.html#map-style-datasets
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#%%
class WineDataset(Dataset):
    def __init__(self):
        xy= np.loadtxt('./data/wine/wine.csv', delimiter= ',', skiprows= 1, dtype = np.float32)
        self.x= torch.from_numpy(xy[:,1:])
        self.y= torch.from_numpy(xy[:, [0]])
        self.n_samples= xy.shape[0]




    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return self.n_samples

#%%
dataset= WineDataset()
first_data= dataset[0]
features, labels= first_data
batch_size= 4
dataloader= DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True )


dataiter= iter(dataloader)
data= dataiter.next()
feature, labels= data

print(feature, labels)
total_samples= len(dataset)
n_iterations= np.ceil(total_samples/batch_size)

for epoch in range(2):
    for i, (inputs, lables) in enumerate(dataloader):
        #forward, backward, update
        if (i+1)%5== 0: 
            print(f"epoch {epoch} , step {i+1}/{n_iterations}, inputs {inputs.shape}")



# %%
