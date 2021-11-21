import torch
from torch.nn.modules.container import ParameterDict
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#%%
train_dataset= torchvision.datasets.CIFAR10(root= './data/', train= True, transform= torchvision.transforms.transforms.ToTensor(), download= True)
test_dataset= torchvision.datasets.CIFAR10(root= './data/', train= False, transform= torchvision.transforms.transforms.ToTensor())
print(train_dataset.class_to_idx)
#%%
batch_size= 6


train_loader= torch.utils.data.DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle= True)
train_loader= torch.utils.data.DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle= False)

example = iter(train_loader)
samples, labels= example.next()
print(samples.shape, labels.shape)


for i in range(batch_size):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0])


# %%
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size= 1024 # 28*28 
hidden_size= np.arange([120, 84])
num_classes= 10
num_epochs= 2
lr= 0.001

#%%
#define the model

class CNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super().__init__(input_size,num_classes)
        self.input_size= input_size
        self.conv1= torch.nn.Conv2d(in_channels=3, out_channels= 6, kernel_size= 5)
        self.pool= torch.nn.MaxPool2d(kernle_size= 2, stride= 2)
        self.conv2= torch.nn.Conv2d(in_channels=6, out_channels= 16, kernel_size= 5)
        self.fc1= torch.nn.Linear(16*5*5, hidden_size[0])
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = torch.nn.Linear(hidden_size[1], num_classes)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = torch.nn.functional.relu(self.fc1(x))               # -> n, 120
        x = torch.nn.functional.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

model = ConvNet().to(device)

criterion= torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= lr)
