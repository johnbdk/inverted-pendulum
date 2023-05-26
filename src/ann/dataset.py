import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DiskDataset(Dataset):
    def __init__(self, file, transform = None):
        self.data = pd.read_csv(file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        u = torch.tensor(self.data.iloc[index, 0])
        th = torch.tensor(self.data.iloc[index, 1])

        if self.transform:
            item = self.transform(item)

        return [u, th]

dataset_train = DiskDataset(file = "../data/training-data.csv")
u_list_train = []
th_list_train = []
for i in range(len(dataset_train)):
    u, th = dataset_train[i]
    u_list_train.append(u.item())  # Convert tensor to a scalar value
    th_list_train.append(th.item())  # Convert tensor to a scalar value

