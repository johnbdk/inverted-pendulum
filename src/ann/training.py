import numpy as np
import torch

def train(model, train_dataloader, validation_dataloader, loss_fcn, optimizer, epochs):
    loss_data = []
    v_loss_data = []
    for epoch in range(epochs):
        loss = 0
        v_loss = 0 
        for batch,_ in train_dataloader:
            
            optimizer.zero_grad()
            batch=batch.type(torch.DoubleTensor)
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


    




