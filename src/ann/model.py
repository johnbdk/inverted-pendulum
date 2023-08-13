# System imports
import os

# External Imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import itertools 

# Local Relative Imports
from config.definitions import *
from ann.architectures import NARXNet, NOENet
from dataset.processing import DiskDataset, split, normalize

def train_narx(na, nb, hidden_nodes):
    
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file=dataset_name, na=na, nb=nb)

    # Create dataloaders
    train_dataset, test_dataset = split(dataset, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    # Evaluation dataloader for the final presentation of results
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    epochs = 1

    # Create model with given hyperparameters
    model = NARXNet(na + nb, hidden_nodes)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fcn = torch.nn.MSELoss()
    def nrms_loss(output, target):
        return torch.sqrt(torch.mean((output - target)**2)) / (torch.max(target) - torch.min(target))

    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(epochs):
        t_loss = 0
        for u, th in train_dataloader:
            outputs = model(u)
            th_s = torch.squeeze(th)
            
            # Calculate loss
            train_Loss = nrms_loss(outputs, th_s)
            optimizer.zero_grad()
            train_Loss.backward()
            optimizer.step()
            t_loss = t_loss + train_Loss.item()

        # Calculate average loss for training samples
        t_loss = t_loss / len(train_dataloader)
        
        # Calculate loss for the test dataset
        val_loss = 0
        for u, th in test_dataloader:
            outputs = model(u)
            th_s = torch.squeeze(th)
            val_Loss = nrms_loss(outputs, th_s)
            val_loss = val_loss + val_Loss.item()
        val_loss = val_loss / len(test_dataloader)

        print("Epoch: %d, Training NRMS Loss: %f, Validation NRMS Loss: %f" % (epoch + 1, t_loss, val_loss))
        epoch_train_loss.append(t_loss)
        epoch_val_loss.append(val_Loss)

    torch.save(model.state_dict(), os.path.join(GRID_DIR, 'Network_NARX_'+str(na)+'_'+str(nb)+'.pth'))

def train_narx_grid():
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    
    # Values to be searched
    na_list = [2, 3, 4, 5, 6, 7]
    nb_list = [2, 3, 4, 5, 6, 7]

    size = (len(na_list), len(nb_list))
    final_score_t = np.zeros(size)
    final_score_val = np.zeros(size)

    for i, na in enumerate(na_list):
        for j, nb in enumerate(nb_list):
            # Load dataset
            dataset = DiskDataset(file=dataset_name, na=na, nb=nb)

            # Create dataloaders
            train_dataset, test_dataset = split(dataset, 0.8)
            train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

            epochs = 25

            # Create model with given hyperparameters
            model = NARXNet(na + nb, 32)
            print(model)
            optimizer = torch.optim.Adam(model.parameters())

            def nrms_loss(output, target):
                return torch.sqrt(torch.mean((output - target)**2)) / (torch.max(target) - torch.min(target))

            epoch_train_loss = []
            epoch_val_loss = []
            for epoch in range(epochs):
                t_loss = 0
                for u, th in train_dataloader:
                    outputs = model(u)
                    th_s = torch.squeeze(th)
                    
                    # Calculate loss
                    train_Loss = nrms_loss(outputs, th_s)
                    optimizer.zero_grad()
                    train_Loss.backward()
                    optimizer.step()
                    t_loss = t_loss + train_Loss.item()

                # Calculate average loss for training samples
                t_loss = t_loss / len(train_dataloader)
                
                # Calculate loss for the test dataset
                val_loss = 0
                for u, th in test_dataloader:
                    outputs = model(u)
                    th_s = torch.squeeze(th)
                    val_Loss = nrms_loss(outputs, th_s)
                    val_loss = val_loss + val_Loss.item()
                val_loss = val_loss / len(test_dataloader)

                print("Epoch: %d, Training NRMS Loss: %f, Validation NRMS Loss: %f" % (epoch + 1, t_loss, val_loss))
                epoch_train_loss.append(t_loss)
                epoch_val_loss.append(val_loss)
            final_score_t[i][j] = t_loss
            final_score_val[i][j] = val_loss
            np.savez('grid_search_ann/Train_loss_' + str(na) + '_' +str(nb), train_loss = epoch_train_loss, test_loss = epoch_val_loss)
            torch.save(model.state_dict(), 'grid_search_ann/Network_NARX_' + str(na) + '_' + str(nb) + '.pth')

    np.savez('grid_search_ann/final_test_nrms', score = final_score_t)
    np.savez('grid_search_ann/final_val_nrms', score = final_score_val)

def eval_narx(na, nb):
    # Create path for checkpoint file
    file = os.path.join(GRID_DIR, 'Network_NARX_'+str(na)+'_'+str(nb)+'.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    hidden_nodes = 32
    model = NARXNet(na + nb, hidden_nodes)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    
    dataset = DiskDataset(file=dataset_name, na=na, nb=nb)
    _, dataset = split(dataset, 0.7)
    # Evaluation dataloader for the final presentation of results
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run the prediction evaluation ##############################
    eval_out =[]
    eval_pred = []
    eval_error = []
    with torch.no_grad():
        for u, th in eval_dataloader:
            eval_out.append(th.detach().numpy())
            out = model(u)
            eval_pred.append(out.detach().numpy())
            eval_error.append((out.detach().numpy()-th.detach().numpy()))
    
    eval_out = np.reshape(eval_out, [len(eval_out), 1])
    eval_out = eval_out*0.4904721616946861+0.034056081732066625
    eval_pred = np.reshape(eval_pred, (len(eval_pred), 1))
    eval_pred = eval_pred*0.4904721616946861+0.034056081732066625
    eval_error = np.reshape(eval_error, (len(eval_error), 1))

    for length in [800, 8000]:
        # Figure with comparison to ground truth
        plt.figure(length)
        plt.subplot(2,1,1)
        time = np.arange(length)
        plt.plot(eval_pred[0:length])
        plt.plot(eval_out[0:length],'--')
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        plt.legend(["Actual", "Predicted"])
        plt.title("ANN NARX prediction")
        plt.grid()

        # Figure with error
        plt.subplot(2,1,2)
        
        plt.plot(eval_out[0:length])
        plt.plot(eval_error[0:length])
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        plt.legend(["Ground truth angle", "Residual angle error"])
        plt.title("Prediction error")
        plt.grid()
        plt.show()

    # Run the simulation evaluation #####################################
    sim_out =[]
    sim_pred = []
    sim_error = []

    with torch.no_grad():
        for i, (u, th) in enumerate(eval_dataloader, 0):
            sim_out.append(th.detach().numpy())
            if i<=10:
                out = model(u)
                sim_pred.append(out.detach().numpy().item())
                sim_error.append((out.detach().numpy()-th.detach().numpy()))
            else:
                sim_u = np.concatenate((u[0][0:na],(np.squeeze(sim_pred[i-nb:i]))),0)
                sim_u = torch.DoubleTensor(sim_u)
                out = model(torch.unsqueeze(sim_u, 0))
                out = out.detach().numpy()
                out = np.squeeze(out)
                sim_pred.append(out.item())
                sim_error.append((out-th.detach().numpy()))
    
    sim_out=np.squeeze(sim_out)
    sim_pred = np.squeeze(sim_pred)
    sim_error = np.squeeze(sim_error)
    sim_out = sim_out*0.4904721616946861+0.034056081732066625
    sim_pred = sim_pred*0.4904721616946861+0.034056081732066625
    sim_error = sim_out-sim_pred
    for length in [800, 8000]:
        # Figure with comparison to ground truth
        plt.figure(length)
        plt.subplot(2,1,1)
        time = np.arange(length)
        plt.plot(sim_pred[0:length])
        plt.plot(sim_out[0:length],'--')
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        plt.legend(["Actual", "Predicted"])
        plt.title("ANN NARX prediction")
        plt.grid()

        # Figure with error
        plt.subplot(2,1,2)
        
        plt.plot(sim_out[0:length])
        plt.plot(sim_error[0:length])
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        plt.legend(["Ground truth angle", "Residual angle error"])
        plt.title("Prediction error")
        plt.grid()
        plt.show()


    return np.average(eval_error), np.average(sim_error)

def train_noe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file=dataset_name, na=40, nb=0, nc=37)

    train_dataset, test_dataset = split(dataset, 0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)


    epochs = 15
    n_burn = 10
    hidden_size = 32
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)
    file = os.path.join(MODELS_DIR, 'Network_NARX_2_2.pth')

    state_dict = torch.load(file)
    with torch.no_grad():
        model.lay1.weight.copy_(state_dict['lay1.weight'])
        model.lay1.bias.copy_(state_dict['lay1.bias'])
        model.lay2.weight.copy_(state_dict['lay2.weight'])
        model.lay2.bias.copy_(state_dict['lay2.bias'])


    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(epochs):
        t_loss = 0
        for u, th in train_dataloader:
            th = th[:, 1:]
            outputs = model(u)
            train_Loss = torch.mean((outputs - th)[:, n_burn:] ** 2)
            optimizer.zero_grad()
            train_Loss.backward()
            optimizer.step()
            t_loss = t_loss + train_Loss.item()
        t_loss = t_loss / (len(train_dataloader) - n_burn)
        val_loss = 0

        for u, th in test_dataloader:
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
        th = th[:,1:]
        eval_out.append(th.detach().numpy())  # Convert eval_out to a PyTorch tensor
        out = model(u)
        eval_pred.append(out.detach().numpy())
        print(np.shape(eval_out))
        print(np.shape(eval_pred))
    eval_out = np.squeeze(eval_out)
    eval_pred = np.squeeze(eval_pred)
    with torch.no_grad():
        print(np.shape(eval_out))
        plt.plot(eval_out[:,-1])
        plt.plot(eval_pred[:,-1])
        plt.xlabel('k')
        plt.ylabel('y')
        plt.xlim(0,80000)
        plt.legend(['real','predicted'])
        plt.show()

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'Network_NOE_2_2.pth'))

def eval_noe():
    # Create path for checkpoint file
    file = os.path.join(MODELS_DIR, 'Network_NOE_2_2.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    hidden_size = 32
    na = 2 
    nb = 2
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    
    dataset = DiskDataset(file=dataset_name, na=40, nb=0, nc=37)
    dataset, ddataset = split(dataset, 0.8)
    # Evaluation dataloader for the final presentation of results

    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    # Run the evaluation
    eval_out =[]
    eval_pred = []
    eval_error = []
    for u, th in eval_dataloader:
        th = th[:,1:]
        eval_out.append(th.detach().numpy())  # Convert eval_out to a PyTorch tensor
        out = model(u)
        eval_pred.append(out.detach().numpy())
    eval_out = np.squeeze(eval_out)
    eval_pred = np.squeeze(eval_pred)
    
    # Figure with comparison to ground truth
    plt.figure(1)
    plt.subplot(2,1,1)
    length = 7990
    time = np.arange(length)
    # eval_out = np.reshape(eval_out, [len(eval_out), 1])
    eval_out = eval_out*0.4904721616946861+0.034056081732066625
   
    # eval_pred = np.reshape(eval_pred, (len(eval_pred), 1))
    eval_pred = eval_pred*0.4904721616946861+0.034056081732066625
    eval_error = eval_out[:,-1] - eval_pred[:,-1]
    plt.plot(eval_pred[:,-1][0:length])
    plt.plot(eval_out[:,-1][0:length],'--')
    # plt.errorbar(time, eval_pred[0:length], yerr=2*var, fmt='.r', label='Estimated angle')
    # plt.xlim(0, length)
    plt.xlabel('time')
    plt.ylabel('Theta angle')
    plt.legend(["Actual", "Predicted"])
    plt.title("ANN NOE prediction")
    plt.grid()

    

    # Figure with error
    plt.subplot(2,1,2)
    eval_error = np.reshape(eval_error, (len(eval_error), 1))
    plt.plot(eval_out[:,-1][0:length])
    plt.plot(eval_error[0:length])
    # plt.xlim(0, length)
    plt.xlabel('time')
    plt.ylabel('Theta angle')
    plt.legend(["Ground truth angle", "Residual angle error"])
    plt.title("Prediction error")
    plt.grid()
    plt.show()

    
def simulation_narx():
    # Create path for checkpoint file
    file = os.path.join(MODELS_DIR, 'Network_NARX_2_2.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    na = 2
    nb = 2
    hidden_nodes = 32
    model = NARXNet(na + nb, hidden_nodes)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "test-simulation-submission-file.csv"
    ftrain = "training-data.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    train_dataset_name = os.path.join(DATASET_DIR, ftrain)
    dataset = DiskDataset(file=dataset_name, na=na, nb=nb)
    train_dataset = DiskDataset(file=train_dataset_name, na=na, nb=nb)

    # Evaluation dataloader for the final presentation of results
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run the evaluation
    eval_out =[]
    eval_pred = []
    eval_error = []
    th_test_sim = []
    u_test = []
    # First sequence
    with torch.no_grad():
        for i, (u, th) in enumerate(eval_dataloader, 0):
            u_test.append(u[0][0].detach().numpy()*1.7840869216420139 - 0.0003188249613298902)
            eval_out.append(th.detach().numpy())
            if i<=6:
                out = model(u)
                eval_pred.append(out.detach().numpy().item())
                eval_error.append((out.detach().numpy()-th.detach().numpy())/th.detach().numpy())
                th_test_sim.append(np.squeeze(u[0][na].detach().numpy()*0.4904721616946861+0.034056081732066625))
            else:
                sim_u = np.concatenate((u[0][0:na],(np.squeeze(eval_pred[i-nb:i]))),0)
                th_test_sim.append(np.squeeze(eval_pred[i-4]*0.4904721616946861+0.034056081732066625))
                sim_u = torch.DoubleTensor(sim_u)
                out = model(torch.unsqueeze(sim_u, 0))
                out = out.detach().numpy()
                out = np.squeeze(out)
                eval_pred.append(out.item())

    # Figure with comparison to ground truth
    plt.figure(1)
    length = 50
    eval_out = np.reshape(eval_out, [len(eval_out), 1])
    plt.plot(eval_out[0:length])
    eval_pred = np.reshape(eval_pred, (len(eval_pred), 1))
    plt.plot(eval_pred[0:length],'--')
    plt.xlim(0, length)
    plt.legend(["Actual", "Predicted"])
    plt.title("Predicted and actual 'th'")

    np.savez('test-simulation-example-submission-file.npz', th=th_test_sim, u=u_test)
    
def simulation_noe():
    # Create path for checkpoint file
    file = os.path.join(MODELS_DIR, 'Network_NOE_2_2.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    hidden_size = 32
    na = 2
    nb= 2
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)
    print(model)
    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "test-simulation-submission-file.csv"
    dataset_name = os.path.join(DATASET_DIR, fname)
    dataset = DiskDataset(file=dataset_name, na=40, nb=0, nc=39)

    # Evaluation dataloader for the final presentation of results
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())

    # Run the evaluation
    eval_out =[]
    eval_pred = []
    eval_error = []
    th_test_sim = []
    u_test = []
    # First sequence
    with torch.no_grad():
        for u, th in eval_dataloader:
            th = th[:,1:]
            eval_out.append(th.detach().numpy())  # Convert eval_out to a PyTorch tensor
            out = model(u)
            eval_pred.append(out.detach().numpy())
        eval_out = np.squeeze(eval_out)
        eval_pred = np.squeeze(eval_pred)
        th_test_sim = eval_pred[:,-1]
        u_test = u[:,0].detach().numpy()

    np.savez('test-simulation-example-submission-file.npz', th=th_test_sim, u=u_test)

def prediction_narx():
    # Create path for checkpoint file
    file = os.path.join(MODELS_DIR, 'Network_NARX_2_2.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    na = 2
    nb = 2
    hidden_nodes = 32
    model = NARXNet(na + nb, hidden_nodes)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "test-prediction-submission-file.npz"
    dataset_name = os.path.join(DATASET_DIR, fname)

    data = np.load(dataset_name)
    upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
    thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
    upast_test_un = upast_test
    thpast_test_un = thpast_test
    upast_test = (upast_test + 0.0003188249613298902)/1.7840869216420139
    thpast_test = (thpast_test-0.034056081732066625)/0.4904721616946861

    Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)
    Xtest = torch.DoubleTensor(Xtest)
    Ypredict = model(Xtest)
    Ypredict = (Ypredict.detach().numpy()-0.034056081732066625)/0.4904721616946861
    np.savez('test-prediction-example-submission-file.npz', upast=upast_test_un, thpast=thpast_test_un, thnow=Ypredict)

def prediction_noe():
    
    # Create path for checkpoint file
    file = os.path.join(MODELS_DIR, 'Network_NOE_2_2.pth')
    if not os.path.isfile(file):
        print(file)
        print("no file")
        return 0
    
    # Create model
    hidden_size = 32
    na = 2
    nb = 2
    input_size = 2
    s_size = 2
    output_size = 1
    model = NOENet(hidden_size, input_size, s_size, output_size)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(file)) 

    # Load dataset
    fname = "test-prediction-submission-file.npz"
    dataset_name = os.path.join(DATASET_DIR, fname)

    data = np.load(dataset_name)
    upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
    thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
    upast_test_un = upast_test
    thpast_test_un = thpast_test
    upast_test = (upast_test + 0.0003188249613298902)/1.7840869216420139
    thpast_test = (thpast_test-0.034056081732066625)/0.4904721616946861

    Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)
    Xtest = torch.DoubleTensor(Xtest)
    Ypredict = model(Xtest)[:,-1]
    Ypredict = (Ypredict.detach().numpy()-0.034056081732066625)/0.4904721616946861
    np.savez('test-prediction-example-submission-file.npz', upast=upast_test_un, thpast=thpast_test_un, thnow=Ypredict)

def eval_grid():
    na_list = [2, 3, 4, 5, 6]
    nb_list = [2, 3, 4, 5, 6]
    pred_err = np.zeros((len(na_list), len(nb_list)))
    sim_err = np.zeros((len(na_list), len(nb_list)))
    for i, na in enumerate(na_list):
        for j, nb in enumerate(nb_list):
            pred_err[i][j], sim_err[i][j] = eval_narx(na, nb)
            # break


    # pred_err = [[-0.00115165, 0.00854777, 0.00421017, 0.00265272, 0.00161407],[0.00511587, 0.0063454, 0.00690616, 0.00419539, 0.00180429],[0.00474716, 0.00680818, 0.00721762, 0.00446633, 0.00413703],[0.00342311, 0.00479204, 0.00614639, 0.00590124, 0.00150929],[0.00373739, 0.00347618, 0.00452532, 0.00359318, 0.00397195]]
    # sim_err = [[-0.01730917, 0.06864505, 0.04395969, 0.02991583, 0.01623182],[0.0598835, 0.04630451, 0.05096129, 0.0307783, 0.0134732],[0.05416022, 0.05121506, 0.04081443, 0.04370086, 0.02716826],[0.03991999, 0.03587696, 0.03849971, 0.04223643, 0.01886482],[0.05551941, 0.02583583, 0.02569334, 0.02420556, 0.09108772]]

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(pred_err)
    plt.xlabel("na")
    plt.ylabel("nb")
    plt.colorbar()
    plt.title("Prediction error")
    plt.xticks([0, 1, 2, 3, 4],na_list)
    plt.yticks([0, 1, 2, 3, 4],nb_list)
    
    
    plt.subplot(1,2,2)
    plt.imshow(sim_err)
    plt.xlabel("na")
    plt.ylabel("nb")
    plt.title("Simulation error")
    plt.xticks([0, 1, 2, 3, 4],na_list)
    plt.yticks([0, 1, 2, 3, 4],nb_list)
    plt.colorbar()
    plt.show()