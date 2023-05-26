from torch import nn
import torch

class Network_NARX(nn.Module): 
    def __init__(self, **kwargs): 
        # invoke nn.Module constructor
        super(Network_NARX,self).__init__() 

        self.in_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=)
        self.out_layer = nn.Linear(in_features=, out_features=)
    
    def forward(self, input): 
        output = input
        return output
    
class Network_NOE(nn.Module): 
    def __init__(self, **kwargs): 
        # invoke nn.Module constructor
        super(Network_NOE,self).__init__() 

        self.in_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=)
        self.out_layer = nn.Linear(in_features=, out_features=)
    
    def forward(self, input): 
        output = input
        return output

class Network_states(nn.Module): 
    def __init__(self, **kwargs): 
        # invoke nn.Module constructor
        super(Network_NOE,self).__init__() 

        self.in_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=)
        self.out_layer = nn.Linear(in_features=, out_features=)
    
    def forward(self, input): 
        output = input
        return output

