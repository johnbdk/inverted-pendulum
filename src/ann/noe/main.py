import torch
from models import Network_NOE
from training import train

model = Network_NOE()

torch.save(model.state_dict(), 'Network_NOE.pth')