import torch
from torch.utils.data import random_split, DataLoader
from models import RNN
from training import train
from dataset import DiskDataset
import torch.optim as optim
import numpy as np

def make_OE_data(udata, ydata, nf=100):
    U = [] 
    Y = [] 
    for k in range(nf, len(udata) + 1):
        U.append(udata[k - nf:k])
        Y.append(ydata[k - nf:k])
    return np.array(U), np.array(Y)

BATCH_SIZE = 1

# Load dataset
dataset = DiskDataset(file="../data/training-data.csv")
dataset_size = len(dataset)
train_size = int(0.5 * dataset_size)  # 50% for training
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = RNN()
model.double()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fcn = torch.nn.MSELoss()

nfuture = 30
convert = lambda x: [torch.tensor(xi, dtype=torch.float64) for xi in x]

dataset_train = DiskDataset(file="../data/training-data.csv")
train_dataset, test_dataset = random_split(dataset_train, [train_size, test_size])

utrain, ytrain = dataset.get_data()

utest, ytest = dataset.get_data()

Utrain, Ytrain = convert(make_OE_data(utrain, ytrain, nf=nfuture))
Utest, Ytest = convert(make_OE_data(utest, ytest, nf=nfuture))


torch.save(model.state_dict(), 'Network_states.pth')
