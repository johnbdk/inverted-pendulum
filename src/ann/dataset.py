import pandas as pd
import torch
from torch.utils.data import random_split, Dataset
import matplotlib.pyplot as plt
import numpy as np

PATH = "data/training-data.csv"

class DiskDataset(Dataset):
    def __init__(self, file, na, nb, transform = None):
        self.data = pd.read_csv(file)
        self.transform = transform
        self.na = na
        self.nb = nb
        self.max = max(self.na, self.nb)
    
    def __len__(self):
        return len(self.data) - self.max
    
    def __getitem__(self, index):
        # u = torch.tensor(np.concatenate([self.data.iloc[index, 0], self.data.iloc[index, 1]]))
        index = index + self.max
        th = torch.tensor(self.data.iloc[index, 1])
        data_u = self.data.iloc[index-self.na:index, 0]
        data_th = self.data.iloc[index-self.nb:index, 1]
        data_u.reset_index(drop=True, inplace=True)
        data_th.reset_index(drop=True, inplace=True)
        u = torch.tensor(np.concatenate([data_u, data_th]))
        
        # u = torch.tensor([self.data.iloc[index, 0], self.data.iloc[index, 1]])

        if self.transform:
            item = self.transform(item)

        return [u, th]

# dataset = DiskDataset(PATH,2,3)
# len = dataset.__len__()
# print(len)
# print(dataset.__getitem__(0))
# print(dataset.__getitem__(len-1))

def split(dataset, percent):
    dataset_size = len(dataset)
    train_size = int(percent * dataset_size)  # 90% for training
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


