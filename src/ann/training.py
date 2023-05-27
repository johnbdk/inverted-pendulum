import numpy as np
import matplotlib.pyplot as plt

def train_NARX(model, train_dataloader, test_dataloader, loss_fcn, optimizer, epochs):

    epoch_loss = []

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

        print(epoch,t_loss)
        epoch_loss.append(t_loss)

        # TO-DO validation

    test_loss = []

    for u, th in test_dataloader:
        outputs = model(u)
        test_loss.append(loss_fcn(outputs, th).item())

    plt.plot(epoch_loss)
    plt.show()

        

def train(model, train_dataloader, validation_dataloader, loss_fcn, optimizer, epochs):
    loss_data = []
    v_loss_data = []
    for epoch in range(epochs):
        loss = 0
        v_loss = 0 
        for batch,_ in train_dataloader:
            
            optimizer.zero_grad()

            outputs = model(batch)
            
            train_loss = loss_fcn(outputs, batch)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()
            
        for features,_ in validation_dataloader:
                v_outputs = model(features)
                v_loss = loss_fcn(v_outputs, features)
                v_loss = v_loss.item()
        # compute the epoch training loss
        loss = loss / len(train_dataloader)

        # Append losses
        loss_data.append(loss)
        v_loss_data.append(v_loss)

        # display the epoch training and validation loss
        print("epoch : {}/{}, loss = {:.6f}, validation loss = {:.6f}".format(epoch + 1, epochs, loss, v_loss))

    return loss_data, v_loss_data


    




