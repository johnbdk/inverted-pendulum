import torch
from models import Network_states
from training import train

model = Network_states()

torch.save(model.state_dict(), 'Network_states.pth')