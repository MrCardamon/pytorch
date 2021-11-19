import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

bc = datasets.load_breast_cancer()
X,y= bc.data, bc.targets

n_samples, n_features= X.shape
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state= 42)

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

X_train= torch.from_numpy(X_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
y_train= torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
y_test= torch.from_numpy(y_test.astype(np.float32)).view(-1,1)
 
#model= torch.nn.Lo