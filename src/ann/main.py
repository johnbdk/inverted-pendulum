import torch
from torch.utils.data import random_split, DataLoader
from models import RNN
from training import train
from dataset import DiskDataset
import torch.optim as optim

BATCH_SIZE = 1

# Load dataset
dataset = DiskDataset(file = "data/training-data.csv")
dataset_size = len(dataset)
train_size = int(0.5 * dataset_size)  # 50% for training
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE)

model = RNN() 
model.double()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fcn = torch.nn.MSELoss()
train(model, train_dataloader,test_dataloader,loss_fcn, optimizer, epochs=1)
# torch.save(model.state_dict(), 'Network_states.pth')