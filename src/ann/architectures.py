# External Imports
import torch
from torch import nn
import numpy as np

class NARXNet(nn.Module): 
    def __init__(self, n_in, n_hidden_nodes): 
        super(NARXNet,self).__init__() 
        self.lay1 = nn.Linear(n_in,n_hidden_nodes).double()
        self.lay2 = nn.Linear(n_hidden_nodes,1).double()
    
    def forward(self,x): 
        #x = concatenated [upast and ypast] 
        x1 = torch.sigmoid(self.lay1(x)) 
        y = self.lay2(x1)[:,0] 
        return y 
    
class NOENet(nn.Module): 
    def __init__(self, hidden_size=20, u_size=2, s_size=2, out_size=1): 
        # invoke nn.Module constructor
        super(NOENet, self).__init__() 
        self.u_size = u_size
        self.s_size = s_size
        self.output_size = out_size
        self.hidden_size = hidden_size

        self.lay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
        self.lay1_bn = nn.BatchNorm1d(self.hidden_size)  # Batch normalization after the first linear layer
        self.lay2 = nn.Linear(self.hidden_size, 1)

        self.rlay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
        self.rlay1_bn = nn.BatchNorm1d(self.hidden_size)  # Batch normalization after the second linear layer
        self.rlay2 = nn.Linear(self.hidden_size, self.s_size)

        # Make it double
        self.double()

    def forward(self, input):
        # batches, time
        output = []

        # h_0
        h = torch.zeros(input.shape[0], self.s_size, dtype=torch.float32)

        for t in range(input.shape[1]-1):
            # Prepare the inputs
            inp = input[:, t:t + self.u_size]
            inp = torch.cat((inp, h), dim=1).double()
            leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)  
            out = leaky_relu(self.lay1_bn(self.lay1(inp)))
            out = self.lay2(out)
            h = leaky_relu(self.rlay1_bn(self.rlay1(inp)))
            h = self.rlay2(h)
            output.append(out)
        output = torch.stack(output, dim=1)
        output = torch.squeeze(output)

        return output
