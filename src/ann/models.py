from torch import nn
import torch

class Network_NARX(nn.Module): 
    def __init__(self, n_in, n_hidden_nodes): 
        super(Network_NARX,self).__init__() 
        self.lay1 = nn.Linear(n_in,n_hidden_nodes).double() 
        self.lay2 = nn.Linear(n_hidden_nodes,1).double() 
    
    def forward(self,x): 
        #x = concatenated [upast and ypast] 
        x1 = torch.sigmoid(self.lay1(x)) 
        y = self.lay2(x1)[:,0] 
        return y 
    
    
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
        super(RNN, self).__init__()
        # Initialize your RNN model here
        self.rnn = RNN()

    def forward(self, x):
        # Forward pass of your RNN model
        return self.rnn(x)