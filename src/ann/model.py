# System imports
import os

# External Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import itertools 

# Local Relative Imports
from config.definitions import *
from ann.architectures import NARXNet, NOENet
from dataset.processing import DiskDataset, split, normalize




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

na_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Replace with the values you want to test for NA
nb_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Replace with the values you want to test for NB
hidden_nodes_values = [8 , 16, 20, 32, 64]  # Replace with the values you want to test for hidden nodes

def train_narx(na, nb, hidden_nodes):
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file=dataset_name, na=na, nb=nb)

    train_dataset, test_dataset = split(dataset, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    u_mean, u_std, th_mean, th_std = normalize(train_dataset)

    epochs = 20

    model = NARXNet(na + nb, hidden_nodes)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    
    def nrms_loss(output, target):
        return torch.sqrt(torch.mean((output - target)**2)) / (torch.max(target) - torch.min(target))

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(epochs):
        t_loss = 0
        for u, th in train_dataloader:
            u = (u - u_mean) / u_std
            th = (th - th_mean) / th_std
            outputs = model(u)
            th_s = torch.squeeze(th)
            train_Loss = nrms_loss(outputs, th_s)
            optimizer.zero_grad()
            train_Loss.backward()
            optimizer.step()
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss / len(train_dataloader)
        
        val_loss = 0
        for u, th in test_dataloader:
            u = (u - u_mean) / u_std
            th = (th - th_mean) / th_std
            outputs = model(u)
            th_s = torch.squeeze(th)
            val_Loss = nrms_loss(outputs, th_s)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss / len(test_dataloader)

        print("Epoch: %d, Training NRMS Loss: %f, Validation NRMS Loss: %f" % (epoch + 1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_Loss)

    eval_out =[]
    eval_pred = []
    for u, th in eval_dataloader:
        eval_out.append(th.detach().numpy())
        out = model(u)
        eval_pred.append(out.detach().numpy())

    #eval_out = np.reshape(eval_out, [79998, 1])
    #plt.plot(eval_out)
    #eval_pred = np.reshape(eval_pred, (79998, 1))
    #plt.plot(eval_pred)
    #plt.xlim(0, 8000)
    #plt.show()

    torch.save(model.state_dict(), 'Network_NARX.pth')
    #with torch.no_grad():
        #train_errors = (eval_out - eval_pred) ** 2
        #avg_train_errors = np.mean(train_errors, axis=0) ** 0.5
        #plt.plot(avg_train_errors)
        #plt.title('Batch Averaged Time-Dependent Error')
        #plt.ylabel('Error')
        #plt.xlabel('i')
        #plt.grid()
        #plt.show()

    epoch_val_loss_list = [tensor.detach().numpy() for tensor in epoch_val_loss]

    # Assuming epoch_train_loss is a list
    epoch_train_loss_np = np.array(epoch_train_loss)

    # Plot the data
    #plt.plot(epoch_val_loss_list, label='Validation Loss')
    #plt.plot(epoch_train_loss_np, label='Training Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()

    return epoch_val_loss[-1]  # Return the final validation loss for grid search

def grid_search():
# Perform grid search
    best_val_loss = float('inf')
    best_params = None
    for NA in na_values:
        for NB in nb_values:
            for hidden_nodes in hidden_nodes_values:
                print(f"Training with NA={NA}, NB={NB}, and {hidden_nodes} hidden nodes...")
                val_loss = train_narx(NA, NB, hidden_nodes)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = (NA, NB, hidden_nodes)

    print("Best Hyperparameters: NA=%d, NB=%d, Hidden Nodes=%d, Validation Loss=%.4f" % best_params)


def train_noe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file=dataset_name, na=40, nb=0, nc=39)

    train_dataset, test_dataset = split(dataset, 0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)


    epochs = 1
    n_burn = 10
    hidden_size = 32
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)

    # for retraining
    if 0:
        model.load_state_dict(torch.load('Network_NOE.pth'))

    # For training from scratch:
    else:
        state_dict = torch.load('Network_NARX.pth')
        with torch.no_grad():
            model.lay1.weight.copy_(state_dict['lay1.weight'])
            model.lay1.bias.copy_(state_dict['lay1.bias'])
            model.lay2.weight.copy_(state_dict['lay2.weight'])
            model.lay2.bias.copy_(state_dict['lay2.bias'])

            model.rlay1.weight.copy_(state_dict['lay1.weight'])
            model.rlay1.bias.copy_(state_dict['lay1.bias'])
            model.rlay2.weight.copy_(state_dict['lay2.weight'])
            model.rlay2.bias.copy_(state_dict['lay2.bias'])

    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    

    u_mean, u_std, th_mean, th_std = normalize(train_dataset)
    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(epochs):
        t_loss = 0
        for u, th in train_dataloader:
            th = th[:, 1:]
            u = u - u_mean / u_std
            th = th - th_mean / th_std
            outputs = model(u)
            train_Loss = torch.mean((outputs - th)[:, n_burn:] ** 2)
            optimizer.zero_grad()
            train_Loss.backward()
            optimizer.step()
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss / (len(train_dataloader) - n_burn)
        val_loss = 0

        for u, th in test_dataloader:
            u = u - u_mean / u_std
            th = th - th_mean / th_std
            th = th[:, 1:]
            outputs = model(u)
            val_Loss = torch.mean((outputs - th)[:, n_burn:] ** 2)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss / (len(test_dataloader) - n_burn)

        print("Epoch: %d, Training Loss: %f, Validation Loss: %f" % (epoch + 1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_loss)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    eval_out = []
    eval_pred = []
    for u, th in eval_dataloader:
        eval_out.append(th.detach().numpy())  # Convert eval_out to a PyTorch tensor
        out = model(u)
        eval_pred.append(out.detach().numpy())

    with torch.no_grad():
        plt.plot(eval_out[0])
        plt.plot(eval_pred[0],'--')
        plt.xlabel('k')
        plt.ylabel('y')
        plt.xlim(0,250)
        plt.legend(['real','predicted'])
        plt.show()

    with torch.no_grad():
        train_errors = (eval_out - model(u)) ** 2
        avg_train_errors = torch.mean(train_errors, axis=0) ** 0.5
        plt.plot(avg_train_errors.numpy())
        plt.title('Batch Averaged Time-Dependent Error')
        plt.ylabel('Error')
        plt.xlabel('i')
        plt.grid()
        plt.show()

    # Plot the error per epoch
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    torch.save(model.state_dict(), 'Network_NOE.pth')

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

def eval_noe():
    # Model
    hidden_size = 32
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)

    # Load model
    model.load_state_dict(torch.load('Network_NOE.pth'))

    # Data
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file = dataset_name, na=80, nb=0, nc=79)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())


    eval_out =[]
    eval_pred = []
    for u, th in eval_dataloader:
        th = th[:,1:]
        eval_out.append(th.detach().numpy())
        out = model(u)
        eval_pred.append(out.detach().numpy())


