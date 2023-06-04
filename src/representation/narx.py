# External imports
import torch
import numpy as np
from typing import Callable

# Local imports
from gp.model import PendulumGP, PendulumSGP

class NARX:
    def __init__(self, num_inputs=3, num_outputs=3) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def simulate(self, u: np.ndarray, f : Callable[[list, list], tuple or torch.tensor]) -> dict:
        # Init upast and ypast as lists.
        upast = [0]*self.num_inputs
        ypast = [0]*self.num_outputs
        
        ylist_mean = []
        y_list_var = []

        results = {}
        for unow in u.tolist():
            # Compute the current y given by f
            y_pred = f(np.concatenate((upast, ypast))[None, :])
            # Differentiate output between GP and ANN
            if type(f.__self__) == PendulumGP or type(f.__self__) == PendulumSGP:
                y_pred_mean_np = y_pred[0][0] # numpy of means
                y_pred_var_np = y_pred[1][0] # numpy of vars
                ynow = y_pred_mean_np[0] # take the one and only element of means (since its a simulation)
                y_list_var.append(y_pred_var_np[0])
            else:
                # TODO for ANN
                ynow = y_pred
                
            #update past arrays
            upast.append(unow)
            upast.pop(0)
            ypast.append(ynow)
            ypast.pop(0)
            
            #save result
            ylist_mean.append(ynow)

        results['mean'] = np.array(ylist_mean)
        results['var'] = np.array(y_list_var)
        return results
    
    def make_training_data(self, u : np.ndarray, y : np.ndarray) -> np.ndarray:
        """
            Creates training data from input list and output list using
            the defined system representation.
        """
        io_max = max(self.num_inputs, self.num_outputs)
        Xdata = np.empty(shape=(u.shape[0] - io_max, self.num_inputs+self.num_outputs))
        for i in range(io_max, u.shape[0]):
            Xdata[i - io_max] = np.concatenate([u[i-self.num_inputs:i], y[i-self.num_outputs:i]])
        # print("u shape: {}".format(u.shape))
        # print("y shape: {}".format(y.shape))
        # print("Xdata shape: {}".format(Xdata.shape))
        # print("u[{}:{}]: {}".format(0, self.num_inputs, u[0:self.num_inputs]))
        # print("y[{}:{}]: {}".format(0, self.num_outputs, y[0:self.num_outputs]))
        # print("Xdata[{}]: {}".format(0, Xdata[0]))
        return Xdata, y[io_max:]
    