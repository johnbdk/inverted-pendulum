from torch import nn
import torch

class Network_NARX(nn.Module,): 
    def __init__(self, **kwargs): 
        # invoke nn.Module constructor
        super(Network_NARX,self).__init__() 

        self.in_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=1)
        self.out_layer = nn.Linear(in_features=1 , out_features=1)
    
    def forward(self, input): 
        output = input
        return output
    
class Network_NOE(nn.Module): 
    def __init__(self, **kwargs): 
        # invoke nn.Module constructor
        super(Network_NOE,self).__init__() 

        self.in_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=1)
        self.out_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, input): 
        output = input
        return output

class RNN(nn.Module): 
    def __init__(self): 
        # invoke nn.Module constructor
        super(RNN,self).__init__() 
        self.layer1 = nn.Linear(1, 10, True)
        self.layer2 = nn.Linear(10, 1, True)

    
    def forward(self, inputs):

        output = self.layer1(inputs)
        output = torch.relu(output)
        output = self.layer2(output)
        output = torch.relu(output)
        return output