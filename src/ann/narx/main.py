import torch
from models import Network_NARX
from training import train

model = Network_NARX()

torch.save(model.state_dict(), 'Network_NARX.pth')