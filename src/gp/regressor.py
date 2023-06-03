# External libraries
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import numpy as np


class PendulumGP:
    """This class represents a Gaussian Process model for a Pendulum system."""

    def __init__(self) -> None:
        """
        Constructor for PendulumGP.
        """
        # define kernel
        self.kernel = RBF(length_scale=1.0)

        # define regressor
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)

        print('%s initialized with kernel:%s' % (__class__.__name__, self.kernel))

    def fit(self, X : np.ndarray, Y : np.ndarray) -> None:
        """
        Fit the Gaussian Process model to the data.

        Parameters:
        X (numpy.ndarray): Input variables.
        y (numpy.ndarray): Output variables.
        """
        self.gp.fit(X, Y)
    
    def predict(self, X : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict output variables using the Gaussian Process model.

        Parameters:
        X (numpy.ndarray): Input variables.

        Returns:
        numpy.ndarray: Predicted output variables.
        numpy.ndarray: Standard deviations of the predictions.
        """
        return self.gp.predict(X, return_std=True)

    def plot(self, X : np.ndarray, Y : np.ndarray, Y_pred : np.ndarray, sigma : np.ndarray) -> None:
        """
        Plot the original data, predictions, and confidence intervals.

        Parameters:
        X (numpy.ndarray): Input variables.
        y (numpy.ndarray): Original output variables.
        y_pred (numpy.ndarray): Predicted output variables.
        sigma (numpy.ndarray): Standard deviations of the predictions.
        """

        time = np.arange(0, X.shape[0])

        # Plot original data
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(time, X, label='Input Voltage u (GT)')
        plt.plot(time, Y, label='Output Angle th (GT)')
        plt.xlabel('time')
        plt.ylabel('Input Voltage (u)')
        plt.title('Original Pendulum Data')
        plt.legend()
        plt.grid()
        
        # Plot prediction and error
        plt.subplot(3, 1, 2)
        plt.plot(time, Y, label='Output Angle th (GT)')
        plt.errorbar(time, Y_pred, yerr=2*sigma, fmt='.r')
        plt.xlabel('time')
        plt.ylabel('Output angle (th)')
        plt.title('GP Prediction with error')
        plt.legend()
        plt.grid()

        # Plot error
        plt.subplot(3, 1, 3)
        plt.plot(time, Y-Y_pred, label='Output angle error')
        plt.xlabel('time')
        plt.ylabel('Estimation error (th)')
        plt.title('GP Estimation Error')
        plt.legend()
        plt.grid()

        
        plt.show()