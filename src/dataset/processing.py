# External Imports
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DiskDataset(Dataset):
    def __init__(self, file, na, nb, nc=0, transform = None):
        self.data = pd.read_csv(file)
        self.transform = transform
        self.na = na
        self.nb = nb
        self.nc = nc
        self.max = max(self.na, self.nb)
    
    def __len__(self):
        return len(self.data) - self.max
    
    def __getitem__(self, index):
        # u = torch.tensor(np.concatenate([self.data.iloc[index, 0], self.data.iloc[index, 1]]))
        index = index + self.max
        data_u = self.data.iloc[index-self.na:index, 0]
        data_th = self.data.iloc[index-self.nb:index, 1]
        data_th_out = self.data.iloc[index-self.nc:index+1,1]
        data_u.reset_index(drop=True, inplace=True)
        data_th.reset_index(drop=True, inplace=True)
        data_th_out.reset_index(drop=True, inplace=True)
        u = torch.tensor(np.concatenate([data_u, data_th]))
        th = torch.tensor(data_th_out)
        # u = torch.tensor([self.data.iloc[index, 0], self.data.iloc[index, 1]])

        if self.transform:
            u = self.transform(u)
            th = self.transform(th)

        return [u, th]

dataset = DiskDataset('data/training-data.csv',1,0)
dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle = False)
for u, th in dataloader:
    u_mean, u_std = torch.mean(u), torch.std(u)
    th_mean, th_std = torch.mean(th), torch.std(th)

print(u_mean, u_std)
# len = dataset.__len__()
# print(len)
# print(dataset.__getitem__(0))
# print(dataset.__getitem__(len-1))

def normalize(dataset):
    dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle = False)
    for u, th in dataloader:
        u_mean, u_std = torch.mean(u), torch.std(u)
        th_mean, th_std = torch.mean(th), torch.std(th)
    return u_mean, u_std, th_mean, th_std

def split(dataset, percent):
    dataset_size = len(dataset)
    train_size = int(percent * dataset_size)  # 90% for training
    print(train_size)
    test_size = dataset_size - train_size 
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, train_size + test_size))

    # train_dataset, test_dataset = train_test_split(dataset, test_size, train_size, shuffle=False)
    return train_dataset, test_dataset
