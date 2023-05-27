import torch
from torch.utils.data import DataLoader
from models import Network_NARX
from training import train_NARX
from dataset import DiskDataset, split

BATCH_SIZE = 1000
# series length
na = 2
nb = 2

# Load dataset
dataset = DiskDataset(file = "data/training-data.csv", na=na, nb=nb)

train_dataset, test_dataset = split(dataset, 0.9)

train_dataloader = DataLoader(dataset, BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE)

n_hidden_nodes = 32 
epochs = 10

model = Network_NARX(na+nb, n_hidden_nodes) 
print(model)
optimizer = torch.optim.Adam(model.parameters()) 
loss = torch.nn.MSELoss()
 
train_NARX(model, train_dataloader, test_dataloader, loss, optimizer, epochs)

# torch.save(model.state_dict(), 'Network_states.pth')