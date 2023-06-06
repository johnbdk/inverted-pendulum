import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Step 1: Load and preprocess the npz file
data = np.load("data/training-data.npz")  # Replace 'data.npz' with the path to your npz file
u = data['u']
theta = data['th']

# Step 2: Split the data into training and testing sets
u_train, u_val, theta_train, theta_val = train_test_split(u, theta, test_size=0.5)

u_mean, u_std = np.mean(u_train), np.std(u_train)
theta_mean, theta_std = np.mean(theta_train), np.std(theta_train)

utrain = (u_train - u_mean) / u_std
thetatrain = (theta_train - theta_mean) / theta_std

uval = (u_val - u_mean) / u_std
thetaval = (theta_val - theta_mean) / theta_std

plt.plot(thetatrain)
plt.plot(thetaval)
plt.xlabel('k')
plt.ylabel('y')
plt.show()

import torch

def make_OE_data(udata, ydata, nf=100):
    U = [] #[u[k-nf],...,u[k]]
    Y = [] #[y[k-nf],...,y[k]]
    for k in range(nf, len(udata) + 1):
        U.append(udata[k - nf:k])
        Y.append(ydata[k - nf:k])
    return np.array(U), np.array(Y)

nfuture = 40
convert = lambda x: [torch.tensor(xi, dtype=torch.float64) for xi in x]
Utrain, Thetatrain = convert(make_OE_data(utrain, thetatrain, nf=nfuture))
Uval, Thetaval = convert(make_OE_data(uval, thetaval, nf=len(uval)))

print(Uval.shape)
print(Thetaval.shape)
print(Utrain.shape)  # torch.Size([39961, 40])
print(Thetatrain.shape)  # torch.Size([39961, 40])
class simple_lstm(nn.Module):
    def __init__(self, hidden_size):
        super(simple_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 1
        self.output_size = 1
        net = lambda n_in, n_out: nn.Sequential(nn.Linear(n_in, 40), nn.Sigmoid(), nn.Linear(40, n_out))  # shorthand for a 1 hidden layer NN
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, batch_first=True).double()
        self.h2o = net(hidden_size + self.input_size, self.output_size).double()

    def forward(self, inputs):
        hiddens, (h_n, c_n) = self.lstm(inputs[:, :, None])
        combined = torch.cat((hiddens, inputs[:, :, None]), dim=2)

        h2o_input = combined.view(-1, self.hidden_size + self.input_size)
        y_predict = self.h2o(h2o_input).view(inputs.shape[0], inputs.shape[1])

        return y_predict


n_burn = 20
model = simple_lstm(hidden_size=15)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Use SGD optimizer with 
batch_size = 32
for epoch in range(4):
    for i in range(0, len(Utrain), batch_size):
        Uin = Utrain[i:i+batch_size]
        Yout = model.forward(inputs=Uin)
        Yin = Thetatrain[i:i+batch_size]

        # Apply sigmoid activation to the model's output
        Yout_sigmoid = torch.sigmoid(Yout)

        # Calculate binary cross-entropy loss
        Loss = nn.BCELoss()(Yout_sigmoid[:, n_burn:], Yin[:, n_burn:])
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

    with torch.no_grad():
        Yval_out = model(inputs=Uval)
        Yval_out_sigmoid = torch.sigmoid(Yval_out)
        Loss_val = nn.BCELoss()(Yval_out_sigmoid[:, n_burn:], Thetaval[:, n_burn:])

        Ytrain_out = model(inputs=Utrain)
        Ytrain_out_sigmoid = torch.sigmoid(Ytrain_out)
        Loss_train = nn.BCELoss()(Ytrain_out_sigmoid[:, n_burn:], Thetatrain[:, n_burn:])

        print(f'epoch={epoch}, Validation Loss={Loss_val.item():.2%}, Train Loss={Loss_train.item():.2%}')
with torch.no_grad():
    plt.plot(Thetaval[0])
    plt.plot(model(inputs=Uval)[0],'--')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.xlim(0,250)
    plt.legend(['real','predicted'])
    plt.show()
    plt.plot(np.mean((Thetatrain-model(inputs=Utrain)).numpy()**2,axis=0)**0.5)
    plt.ylabel('error')
    plt.xlabel('time')
    plt.show()
