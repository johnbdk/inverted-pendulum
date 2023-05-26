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
    def __init__(self, hidden_size): 
        # invoke nn.Module constructor
        super(RNN,self).__init__() 
        self.input_size = 1 
        self.output_size = 1
        self.hidden_size = hidden_size
        net = lambda n_in,n_out: nn.Sequential(nn.Linear(n_in,40), \
                                               nn.Sigmoid(), \
                                               nn.Linear(40,n_out)).double() #new short hand
        self.hh2 = net(self.input_size + hidden_size, self.hidden_size) #b=)
        self.h2o = net(self.input_size + hidden_size, self.output_size)
    
    def forward(self, inputs):
       
        hidden = torch.zeros(inputs.shape[0], self.hidden_size, dtype=torch.float64)
        outputs = [] 
        for i in range(inputs.shape[1]): 
            u = inputs[:,i] 
            combined = torch.cat((hidden, u), dim=1) 
            outputs.append(self.h2o(combined)[:,0]) 
            hidden = self.hh2(combined) 
        return torch.stack(outputs,dim=1)