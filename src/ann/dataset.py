import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class DiskDataset(Dataset):
    def __init__(self, file, transform=None):
        self.data = pd.read_csv(file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        u = torch.tensor(self.data.iloc[index, 0]).double()
        th = torch.tensor(self.data.iloc[index, 1]).double()

        if self.transform:
            item = self.transform(item)

        return [u, th]

    def get_data(self):
        u_list = []
        th_list = []
        for i in range(len(self)):
            u, th = self[i]
            u_list.append(u.item())
            th_list.append(th.item())
        return u_list, th_list

dataset_train = DiskDataset(file="../data/training-data.csv")
utrain, ytrain = dataset_train.get_data()
