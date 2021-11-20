import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
bc = datasets.load_breast_cancer()
X,y= bc.data, bc.target 

n_samples, n_features= X.shape
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state= 1234)

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

X_train= torch.from_numpy(X_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
y_train= torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
y_test= torch.from_numpy(y_test.astype(np.float32)).view(-1,1)
 
#model= torch.nn.Lo


class LogisticRegression(torch.nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear= torch.nn.Linear(n_input_features, 1)

    def forward(self,x):
        return torch.sigmoid(self.linear(x))

model= LogisticRegression(n_features)


lr = 0.01
criterion = torch.nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= lr)

for i in range(1000):
    y_hat= model(X_train)
    loss= criterion(y_hat,y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if i%10== 0:
        print(f'iteration {i} error {loss.item():0.4f} ')


with torch.no_grad():
    y_hat= model(X_test).round()
    acc= y_hat.eq(y_test).sum()/y_test.shape[0]
    print(f'accuracy {acc:0.4f}')


# %%
