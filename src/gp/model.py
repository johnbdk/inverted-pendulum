# External libraries
import os
import numpy as np
import pandas as pd


# Local imports
from gp.regressor import PendulumGP, PendulumSGP
from representation.narx import NARX
from config.definitions import *


class PendulumGPManager:
    """This class represents a manager for the Pendulum system.

    The manager is responsible for loading data, training the Gaussian Process (GP) model,
    testing the GP model, and plotting the results.
    """

    def __init__(self, sparse=True, num_inducing_points=0, num_inputs=3, num_outputs=3) -> None:
        """
        Constructor for PendulumGPManager.

        Upon initialization, the manager loads the data, trains the GP model,
        tests the GP model, and plots the results.

        Attributes:
        pendulum_gp (PendulumGP or PendulumSGP): The GP model for the pendulum system.
        """

        # Load dataset
        data = self.load_data(
            file_name=os.path.join(DATASET_DIR, 'training-data.csv'),
            split_ratio=(0.7, 0.2, 0.1)   # 70% train, 20% validation, 10% test
        )

        # Pass data through model representation
        repr = NARX(num_inputs=num_inputs, num_outputs=num_outputs)
        X_train, Y_train = repr.make_training_data(data['train']['X'], data['train']['Y'])
        X_val, Y_val = repr.make_training_data(data['val']['X'], data['val']['Y'])
        X_test, Y_test = repr.make_training_data(data['test']['X'], data['test']['Y'])

        # define the gaussian process model
        if sparse:
            min_x, max_x = np.min(X_train), np.max(X_train)
            inducing_points = np.random.uniform(low=min_x, high=max_x, size=(num_inducing_points, num_inputs + num_outputs))
            self.pendulum_gp = PendulumSGP(X=X_train, Y=Y_train, Z=inducing_points)
        else:
            self.pendulum_gp = PendulumGP()
            

        # train GP model
        self.pendulum_gp.fit(X_train, Y_train)

        # test GP model        
        sim = repr.simulate(data['test']['X'], f=self.pendulum_gp.predict)
        Y_pred = sim['mean']
        var = sim['var']
        print("y_pred.shape", Y_pred.shape)
        # y_pred, sigma = self.pendulum_gp.predict(X_test)

        # plot data
        # self.pendulum_gp.plot(X_test, Y_test, y_pred, sigma)
        self.pendulum_gp.plot(data['test']['X'], data['test']['Y'], Y_pred, var)

    def load_data(self, file_name : str, split_ratio=(0.7, 0.2, 0.1)) -> dict:
        """
        Load data from a CSV file and splits the data into training/validation/test set.

        Parameters:
        file_name (str): Name of the CSV file.
        split_ratio (float): Ratio of train/val/test splits of the dataset.

        Returns:
        numpy.ndarray: Input train variables.
        numpy.ndarray: Output train variables.
        numpy.ndarray: Input val variables.
        numpy.ndarray: Output val variables.
        numpy.ndarray: Input test variables.
        numpy.ndarray: Output test variables.
        """

        # Sanity checks for ratio
        assert 0.0 < split_ratio[0] <= 1.0
        assert 0.0 < split_ratio[1] <= 1.0
        assert 0.0 < split_ratio[2] <= 1.0
        assert np.abs(split_ratio[0] + split_ratio[1] + split_ratio[2] - 1.0) < 1e-3

        # load data
        data = pd.read_csv(file_name)[:5000]
        X = data['u'].values
        Y = data['th'].values

        # load subset of data
        train_samples = int(split_ratio[0] * len(data))
        val_samples = int(split_ratio[1] * len(data))
        test_samples = int(split_ratio[2] * len(data))

        X_train = X[:train_samples]
        Y_train = Y[:train_samples]

        X_val = X[train_samples:train_samples+val_samples]
        Y_val = Y[train_samples:train_samples+val_samples]

        X_test = X[val_samples:val_samples+test_samples]
        Y_test = Y[val_samples:val_samples+test_samples]

        print('Dataset loaded with %d training samples, %d validation samples, %d test samples!' % (train_samples, val_samples, test_samples))
        
        # return data
        return {
            'train' : {'X' : X_train, 
                       'Y' : Y_train},
            'val'   : {'X' : X_val, 
                       'Y' : Y_val},
            'test'  : {'X' : X_test, 
                       'Y' : Y_test}
        }
