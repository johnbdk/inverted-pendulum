import pandas as pd
import torch
from torch.utils.data import Dataset


class DiskDataset(Dataset):
    def __init__(self, file, transform = None):
        self.data = pd.read_csv(file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = torch.tensor(self.data.iloc[index, 0])
        label = torch.tensor(self.data.iloc[index, 1])

        if self.transform:
            item = self.transform(item)

        return [item, label]

