import torch
from torch import nn
from models import Network_NARX
from dataset import DiskDataset 
# choose model
MODEL_PATH = 'Network_NARX.pth'
MODEL_PATH = 'Network_NOE.pth'
MODEL_PATH = 'Network_states.pth'

# data loaders
DiskDataset('../data/training-data.csv')


# loss fcn

loaded_model = Network_NARX(input_shape=784)
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.eval()