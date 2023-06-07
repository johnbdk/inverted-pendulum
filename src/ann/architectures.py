# External Imports
import torch
from torch import nn
import numpy as np

class NARXNet(nn.Module): 
    def __init__(self, n_in, n_hidden_nodes): 
        super(NARXNet,self).__init__() 
        self.lay1 = nn.Linear(n_in,n_hidden_nodes).double()
        self.lay2 = nn.Linear(n_hidden_nodes,1) .double()
    
    def forward(self,x): 
        #x = concatenated [upast and ypast] 
        x1 = torch.sigmoid(self.lay1(x)) 
        y = self.lay2(x1)[:,0] 
        return y 
    
    
class NOENet(nn.Module): 
    def __init__(self, hidden_size): 
        # invoke nn.Module constructor
        super(NOENet,self).__init__() 
        self.hidden_size = hidden_size
        self.in_size = 1
        self.r_size = 20
        self.output_size = 1

        self.net1 = nn.Sequential(nn.Linear(self.in_size + self.hidden_size, self.hidden_size),
                                  nn.Sigmoid(),
                                  nn.Linear(self.hidden_size, 1))
        self.net_r = nn.Sequential(nn.Linear(self.in_size + self.hidden_size, self.hidden_size),
                                  nn.Sigmoid(),
                                  nn.Linear(self.hidden_size, self.hidden_size))

        self.double()
    
    def forward(self, input):
        output = []
        h = torch.zeros(input.shape[0], self.hidden_size, dtype=torch.float32)

        for t in range(input.shape[1]):

            inp = input[:,t]
            inp = inp[:,None]

            inp = torch.cat((inp,h), dim=1).double()
            out = self.net1(inp)
            h = self.net_r(inp)

            output.append(out)

        output = torch.stack(output, dim=1)
        output = torch.squeeze(output)
        # print(np.shape(output))
        return output
    
class simple_RNN(nn.Module):
    def __init__(self, hidden_size):
        super(simple_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 1
        self.output_size = 1
        net = lambda n_in,n_out: nn.Sequential(nn.Linear(n_in,40), \
                                               nn.Sigmoid(), \
                                               nn.Linear(40,n_out)).double() #new short hand
        self.h2h = net(self.input_size + hidden_size, self.hidden_size) #b=)
        self.h2o = net(self.input_size + hidden_size, self.output_size) #b=)
                                                                        #[:,0] should be called after use of h2o
    def forward(self, inputs):
        #input.shape == (N_batch, N_time)
        hidden = torch.zeros(inputs.shape[0], self.hidden_size, dtype=torch.float64) #c)
        outputs = [] #c)
        for i in range(inputs.shape[1]): #c)
            u = inputs[:,i] #shape = (N_batch,) #c)
            combined = torch.cat((hidden, u[:,None]), dim=1) #c) #shape = (N_batch,hidden_size+1)
            outputs.append(self.h2o(combined)[:,0]) #c)
            hidden = self.h2h(combined) #c)
        return torch.stack(outputs,dim=1) #c)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out


class EncoderRNNNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        return output, hidden
