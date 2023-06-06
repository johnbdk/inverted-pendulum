# System imports
import os

# External Imports
import torch
import numpy as np
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

def train_narx():
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file = dataset_name, na=NA, nb=NB)

    train_dataset, test_dataset = split(dataset, 0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=100)
    test_dataloader = DataLoader(test_dataset, batch_size=100)
    eval_dataloader = DataLoader(dataset, batch_size=1)

    n_hidden_nodes = 32 
    epochs = 10

    model = NARXNet(NA+NB, n_hidden_nodes) 
    print(model)
    optimizer = torch.optim.Adam(model.parameters()) 
    loss_fcn = torch.nn.MSELoss()

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(epochs): 
        t_loss = 0
        for u, th in train_dataloader:
            outputs = model(u)
            th_s = torch.squeeze(th)
            # print(np.shape(outputs))
            # print(np.shape(th_s))
            train_Loss = loss_fcn(outputs, th_s)
            # train_Loss = torch.mean((model(batch)-Ytrain)**2) 
            optimizer.zero_grad() 
            train_Loss.backward() 
            optimizer.step()  
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss/len(train_dataloader)
        val_loss = 0
        for u, th in test_dataloader:
            outputs = model(u)
            th_s = torch.squeeze(th)
            val_Loss = loss_fcn(outputs, th_s)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss/len(test_dataloader)
        
        print("Epoch: %d, Training Loss: %f, Validation Loss: %f" % (epoch+1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_Loss)

    eval_out =[]
    eval_pred = []
    for u, th in eval_dataloader:
        eval_out.append(th.detach().numpy())
        out = model(u)
        eval_pred.append(out.detach().numpy())

    eval_out = np.reshape(eval_out, [79998,1])
    plt.plot(eval_out)
    eval_pred = np.reshape(eval_pred, (79998,1))
    plt.plot(eval_pred)
    plt.show()

    torch.save(model.state_dict(), 'Network_NARX.pth')

def train_noe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file = dataset_name, na=40, nb=0, nc=39)

    train_dataset, test_dataset = split(dataset, 0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())

    n_hidden_nodes = 32
    epochs = 10

    model = NOENet(n_hidden_nodes).to(device) 
    print(model)
    optimizer = torch.optim.Adam(model.parameters()) 
    loss_fcn = torch.nn.MSELoss()

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(epochs): 
        t_loss = 0
        for u, th in train_dataloader:
            u, th = u.to(device), th.to(device)
            outputs = model(u)
            # print(u)
            # print(th)
            # print(np.shape(outputs))
            # print(np.shape(th))
            # outputs = torch.squeeze(outputs)
            # print(np.shape(outputs))
            # print('-----------')
            train_Loss = loss_fcn(outputs, th)
            # train_Loss = torch.mean((model(batch)-Ytrain)**2) 
            optimizer.zero_grad() 
            train_Loss.backward() 
            optimizer.step()  
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss/len(train_dataloader)
        val_loss = 0
        for u, th in test_dataloader:
            u, th = u.to(device), th.to(device)
            outputs = model(u)
            val_Loss = loss_fcn(outputs, th)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss/len(test_dataloader)
        
        print("Epoch: %d, Training Loss: %f, Validation Loss: %f" % (epoch+1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_Loss)

    eval_out =[]
    eval_pred = []
    for u, th in eval_dataloader:
        u, th = u.to(device), th.to(device)
        eval_out.append(th.detach().numpy())
        print(np.shape(eval_out))
        out = model(u)
        eval_pred.append(out.detach().numpy())
        print(np.shape(eval_pred))

    eval_out = np.squeeze(eval_out)
    eval_pred = np.squeeze(eval_pred)
    

    plt.plot(eval_out[:,-1])
    # eval_pred = np.reshape(eval_pred, (79999,1))
    plt.plot(eval_pred[:,-1])
    plt.show()

    torch.save(model.state_dict(), 'models/Network_NOE.pth')

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


def eval_ann(fname, model_arch, train_dataloader):
    file = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    checkpoint = load_model(fname)
    # TODO
    if (model_arch == 'narx'):
        model = NARXNet(NA+NB, 32)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        preds = []
        outputs = []
        for u, th in train_dataloader:
            preds.append(model(u))
            outputs.append(th)
        plt.plot(preds)
        plt.plot(outputs)
        plt.show()

    elif (model_arch == 'noe'):
        model = NOENet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    model.load_state_dict(checkpoint)
    return 1

