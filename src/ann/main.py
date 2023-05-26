import torch
from torch.utils.data import random_split
from models import RNN
from training import train
from dataset import DiskDataset
import torch.optim as optim

# Load dataset
dataset = DiskDataset(file = "../data/training-data.csv")
dataset_size = len(dataset)
train_size = int(0.5 * dataset_size)  # 50% for training
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

model = RNN(hidden_size=15) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fcn = torch.nn.MSELoss()
train(model, train_dataset,test_dataset,loss_fcn, optimizer, epochs=10)
torch.save(model.state_dict(), 'Network_states.pth')