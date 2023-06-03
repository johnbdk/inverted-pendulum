# System imports
import os

# External Imports
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Local Relative Imports
from config.definitions import *
from ann.architectures import NARXNet, NOENet
from dataset.processing import DiskDataset, split

def load_model(fname):
    """
    Returns
    ---------
    To use checkpoint after this function you need to define a net arch first, then
    eg. model = NARXNet(); model.load_state_dict(checkpoint['model_state_dict'])
    """
    if (fname == ''):
        raise ValueError("Network architecture name is empty.")
    model_name = os.path.join(MODELS_DIR, fname)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_name, map_location=torch.device(device))
    return checkpoint

def train_ann():
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file = dataset_name, na=NA, nb=NB)

    train_dataset, test_dataset = split(dataset, 0.9)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE)

    n_hidden_nodes = 32 
    epochs = 15

    model = NARXNet(NA+NB, n_hidden_nodes) 
    print(model)
    optimizer = torch.optim.Adam(model.parameters()) 
    loss = torch.nn.MSELoss()
    
    train(model, train_dataloader, test_dataloader, loss, optimizer, epochs)
    torch.save(model.state_dict(), 'Network_NARX.pth')

def train(model, train_dataloader, test_dataloader, loss_fcn, optimizer, epochs):
    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(epochs): 
        t_loss = 0
        for u, th in train_dataloader:
            outputs = model(u)
            train_Loss = loss_fcn(outputs, th)
            # train_Loss = torch.mean((model(batch)-Ytrain)**2) 
            optimizer.zero_grad() 
            train_Loss.backward() 
            optimizer.step()  
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss/len(train_dataloader)
        val_loss = 0
        for u, th in test_dataloader:
            outputs = model(u)
            val_Loss = loss_fcn(outputs, th)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss/len(test_dataloader)
        
        print("Epoch: %d, Training Loss: %f, Validation Loss: %f" % (epoch+1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_Loss)

    # test_loss = []
    # for u, th in test_dataloader:
    #     outputs = model(u)
    #     test_loss.append(loss_fcn(outputs, th).item())

    # plt.plot(epoch_val_loss)
    # plt.plot(epoch_train_loss)
    # plt.show()


def eval_ann(fname, model_arch):
    file = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(file):
        return 0
    
    checkpoint = load_model(fname)
    # TODO
    if (model_arch == 'narx'):
        model = NARXNet()
    elif (model_arch == 'noe'):
        model = NOENet()
    model.load_state_dict(checkpoint)
    return 1

