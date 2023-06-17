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
    def __init__(self, hidden_size = 32, u_size = 2, s_size = 2, out_size = 1): 
        # invoke nn.Module constructor
        super(NOENet,self).__init__() 
        self.u_size = u_size
        self.s_size = s_size
        self.output_size = out_size
        self.hidden_size = hidden_size

        self.lay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
        self.lay2 = nn.Linear(self.hidden_size, 1)

        self.rlay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
        self.rlay2 = nn.Linear(self.hidden_size, self.s_size)
        
        # make it double
        self.double()

    def forward(self, input):
        # batches, time
        output = []

        # h_0
        h = torch.zeros(input.shape[0], self.s_size, dtype=torch.float32)

        for t in range(input.shape[1]-1):

            # prepare the inputs
            inp = input[:, t:t + self.u_size]
            # print("input size", np.shape(inp))
            # print("input", inp)

            inp = torch.cat((inp,h), dim=1).double()

            out = torch.sigmoid((self.lay1(inp)))
            out = self.lay2(out)
            # out = self.narx_net(inp)
            h = torch.sigmoid((self.rlay1(inp)))
            h = self.rlay2(h)
            # h = self.rec_net(inp)
            output.append(out)
        output = torch.stack(output, dim=1)
        output = torch.squeeze(output)

        return output
    
# class NOENet(nn.Module): 
#     def __init__(self, hidden_size = 32, u_size = 2, s_size = 2, out_size = 1): 
#         # invoke nn.Module constructor
#         super(NOENet,self).__init__() 
#         self.u_size = u_size
#         self.s_size = s_size
#         self.output_size = out_size
#         self.hidden_size = hidden_size

#         self.lay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
#         self.lay2 = nn.Linear(self.hidden_size, self.s_size)

#         # self.rlay1 = nn.Linear(self.u_size + self.s_size, self.hidden_size)
#         # self.rlay2 = nn.Linear(self.hidden_size, self.s_size)
        
#         # make it double
#         self.double()

#     def forward(self, input):
#         # batches, time
#         output = []

#         # h_0
#         h = torch.zeros(input.shape[0], self.s_size, dtype=torch.float32)

#         for t in range(input.shape[1]-1):

#             # prepare the inputs
#             inp = input[:, t:t + self.u_size]
#             # print("input size", np.shape(inp))
#             # print("input", inp)

#             inp = torch.cat((inp,h), dim=1).double()

#             out = torch.sigmoid((self.lay1(inp)))
#             out = self.lay2(out)
#             # out = self.narx_net(inp)
#             # print(out)
#             # print(out.size())
#             h = out
#             # h = self.rec_net(inp)
#             output.append(out[:,0])
#         output = torch.stack(output, dim=1)
#         output = torch.squeeze(output)

#         return output
    
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
