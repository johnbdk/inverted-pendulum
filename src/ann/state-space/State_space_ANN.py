import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the npz file
data = np.load("disc-benchmark-files/training-data.npz")  # Replace 'data.npz' with the path to your npz file
u = data['u'].reshape(-1, 1)
theta = data['th'].reshape(-1, 1)

# Normalize the input and output
scaler = MinMaxScaler(feature_range=(-1, 1))
u_normalized = u
theta_normalized = theta

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u_tensor = torch.tensor(u_normalized, dtype=torch.float32).to(device)
theta_tensor = torch.tensor(theta_normalized, dtype=torch.float32).to(device)

# Step 2: Split the data into training and testing sets
utrain, uval, theta_train, theta_test = train_test_split(u_tensor, theta_tensor, test_size=0.5)

# Step 3: Divide the data into batches
batch_size = 32

def divide_batches(input_data, output_data, batch_size):
    num_batches = len(input_data) // batch_size
    input_batches = torch.split(input_data[:num_batches * batch_size], batch_size)
    output_batches = torch.split(output_data[:num_batches * batch_size], batch_size)
    return input_batches, output_batches

utrain_batches, theta_train_batches = divide_batches(utrain, theta_train, batch_size)
uval_batches, theta_test_batches = divide_batches(uval, theta_test, batch_size)

# Step 5: Plot the training and testing data
plt.plot(theta_train)
plt.plot(theta_test)
plt.xlabel('k')
plt.ylabel('y')
plt.show()

def make_OE_data(udata, ydata, nf=100):
    U = []
    Y = []
    for k in range(nf, len(udata) + 1):
        U.append(udata[k - nf:k])
        Y.append(ydata[k - nf:k])

    # Pad the sequences to have consistent lengths
    max_len = max(len(seq) for seq in U + Y)
    U = [np.pad(seq, (0, max_len - len(seq))) for seq in U]
    Y = [np.pad(seq, (0, max_len - len(seq))) for seq in Y]

    return np.array(U), np.array(Y).reshape(-1, max_len)


nfuture = 100
convert = lambda x: [torch.tensor(xi, dtype=torch.float64).to(device) for xi in x]

Utrain, Ytrain = convert(make_OE_data(utrain, theta_train, nf=nfuture))
Uval, Yval = convert(make_OE_data(uval, theta_test, nf=len(uval)))  # uses the whole data set for OE

print(utrain.shape,theta_train.shape, Utrain.shape, Ytrain.shape)



# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 1
        self.output_size = 1
        net = lambda n_in,n_out: nn.Sequential(nn.Linear(n_in,40), \
                                               nn.Sigmoid(), \
                                               nn.Linear(40,n_out)).double() 
        self.hh2 = net(self.input_size + hidden_size, self.hidden_size) 
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
     
input_size = Utrain.shape[2]
hidden_size = 32
output_size = Ytrain.shape[1]
learning_rate = 0.01
num_epochs = 10

# Initialize the RNN model
model = RNN(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(Utrain)
    loss = criterion(outputs, Ytrain.squeeze())
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(Uval)
        val_loss = criterion(val_outputs, Yval)
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Plot the training and validation losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

with torch.no_grad():
    plt.plot(Yval[0])
    plt.plot(model(Uval)[0], '--')  # Remove the 'inputs=' keyword argument
    plt.xlabel('k')
    plt.ylabel('y')
    plt.xlim(0, 250)
    plt.legend(['real', 'predicted'])
    plt.show()

    plt.plot(np.mean((Ytrain - model(Utrain)).numpy()**2, axis=0)**0.5)  # average over the error in batch
    plt.title('batch averaged time-dependent error')
    plt.ylabel('error')
    plt.xlabel('i')
    plt.grid()
    plt.show()
