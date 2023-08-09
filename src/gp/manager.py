# External libraries
import os
import sys
import numpy as np
import pandas as pd

# Local imports
from gp.gp import GaussianProcess, SparseGaussianProcess
from representation.narx import NARX
from config.definitions import DATASET_DIR


class GPManager:
    """This class represents a manager for the Pendulum system.

    The manager is responsible for loading data, training the Gaussian Process (GP) model,
    testing the GP model, and plotting the results.
    """

    def __init__(self, num_inputs=3, num_outputs=3, sparse=True, num_inducing_points=0) -> None:
        """
        Constructor for GPManager.

        Upon initialization, the manager loads the data, trains the GP model,
        tests the GP model, and plots the results.

        Attributes:
        pendulum_gp (GaussianProcess or SparseGaussianProcess): The GP model for the pendulum system.
        """

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.sparse = sparse
        self.num_inducing_points = num_inducing_points

        # Load dataset
        data = self.load_data(
            file_name=os.path.join(DATASET_DIR, 'training-data.csv'),
            split_ratio=(0.7, 0.2, 0.1)   # 70% train, 20% validation, 10% test
        )

        # extract data splits
        X_train, Y_train = data['train']
        X_val, Y_val = data['val']
        X_test, Y_test = data['test']

        # define the gaussian process model
        if sparse:
            min_x, max_x = np.min(X_train), np.max(X_train)
            inducing_points = np.random.uniform(low=min_x, high=max_x, size=(self.num_inducing_points, self.num_inputs + self.num_outputs))
            self.pendulum_gp = SparseGaussianProcess(X=X_train, Y=Y_train, Z=inducing_points, io_max=self.num_inputs + self.num_outputs)
        else:
            self.pendulum_gp = GaussianProcess()
            

        # train GP model
        self.pendulum_gp.fit(X_train, Y_train)

        # test GP model
        sim = repr.simulate(X_test, f=self.pendulum_gp.predict)
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
        data = pd.read_csv(file_name)
        X = data['u'].values
        Y = data['th'].values

        np.set_printoptions(threshold=sys.maxsize)
        print("Number of inputs, outputs: {}, {}".format(self.num_inputs, self.num_outputs))
        print("Data input type: {}, with shape: {}".format(type(X), X.shape))
        print("Data output type: {}, with shape: {}".format(type(Y), Y.shape))

        # # Pass data through model representation
        repr = NARX(num_inputs=self.num_inputs, num_outputs=self.num_outputs)
        X_narx, Y_narx = repr.make_training_data(X, Y)
        print("Features type: {}, with shape: {}".format(type(X_narx), X_narx.shape))
        print("Outputs type: {}, with shape: {}".format(type(Y_narx), Y_narx.shape))


        len_data = X_narx.shape[0]
        # # load subset of data
        train_samples = int(split_ratio[0] * len_data)     # 0.7 * (800000 - max(na,nb)) = 56000
        val_samples = int(split_ratio[1] * len_data)       # 0.2 * (800000 - max(na,nb)) = 16000
        test_samples = int(split_ratio[2] * len_data)      # 0.1 * (800000 - max(na,nb)) = 8000

        X_train = X_narx[:train_samples]
        Y_train = Y_narx[:train_samples]

        X_val = X_narx[train_samples : train_samples+val_samples]
        Y_val = Y_narx[train_samples : train_samples+val_samples]

        X_test = X_narx[val_samples : val_samples+test_samples]
        Y_test = Y_narx[val_samples : val_samples+test_samples]

        print("Train/Val/Test features shapes : {}/{}/{}".format(X_train.shape, X_val.shape, X_test.shape))
        print("Train/Val/Test outputs shapes : {}/{}/{}".format(Y_train.shape, Y_val.shape, Y_test.shape))
        print('Dataset loaded with %d training samples, %d validation samples, %d test samples!' % (train_samples, val_samples, test_samples))

        # return data
        return {
            'train' : (X_train, Y_train),
            'val'   : (X_val, Y_val),
            'test'  : (X_test, Y_test)
        }
