# External libraries
import os
import numpy as np
import pandas as pd


# Local imports
from gp.regressor import PendulumGP
from config.definitions import *


class PendulumManager:
    """This class represents a manager for the Pendulum system.

    The manager is responsible for loading data, training the Gaussian Process (GP) model,
    testing the GP model, and plotting the results.
    """

    def __init__(self) -> None:
        """
        Constructor for PendulumManager.

        Upon initialization, the manager loads the data, trains the GP model,
        tests the GP model, and plots the results.

        Attributes:
        pendulum_gp (PendulumGP): The GP model for the pendulum system.
        """

        # Load dataset
        X, Y = self.load_data(
            file_name=os.path.join(DATASET_DIR, 'training-data.csv'),
            subset_ratio=0.05   # percentage of entire dataset
        )

        # train the gaussian process model
        self.pendulum_gp = PendulumGP()
        self.pendulum_gp.fit(X, Y)

        # test the gaussian process model
        y_pred, sigma = self.pendulum_gp.predict(X)

        # plot data
        self.pendulum_gp.plot(X, Y, y_pred, sigma)


    def load_data(self, file_name : str, subset_ratio=1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Load data from a CSV file.

        Parameters:
        file_name (str): Name of the CSV file.
        subset_ratio (float): Percentage of dataset to load.

        Returns:
        numpy.ndarray: Input variables.
        numpy.ndarray: Output variables.
        """

        # Sanity checks for subset_ratio
        assert 0.0 < subset_ratio <= 1.0

        # load data
        data = pd.read_csv(file_name)

        # load subset of data
        num_samples = int(subset_ratio * len(data))
        X = data['u'].values.reshape(-1, 1)[:num_samples]
        Y = data['th'].values[:num_samples]
        print('Dataset loaded with %d samples!' % len(X))
        
        # return data
        return X, Y
