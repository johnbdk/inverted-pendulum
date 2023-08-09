# External libraries
import numpy as np
from matplotlib import pyplot as plt

# scikit-learn
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

# GPy
import GPy
import GPy.plotting.gpy_plot as gpy_plot

class GaussianProcess:
    """This class represents a Full Gaussian Process model for a Pendulum system."""
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
        y_pred, y_std = self.gp.predict(X, return_std=True)
        return y_pred[:, None], y_std[:, None]

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


class SparseGaussianProcess:
    """This class represents a Sparse Gaussian Process (SGP) model for a Pendulum system."""
    def __init__(self, X : np.ndarray, Y : np.ndarray, Z : np.ndarray, io_max : int) -> None:

        # define kernel
        self.kernel = GPy.kern.RBF(input_dim=io_max, lengthscale=1.0, variance=1.0)

        # define regressor
        if Y.ndim == 1:
            Y = Y[:, None]
        self.gp = GPy.models.SparseGPRegression(X, Y, kernel=self.kernel, Z=Z)

        print('%s initialized with kernel:%s' % (__class__.__name__, self.kernel))

    def fit(self, X : np.ndarray, Y : np.ndarray) -> None:
        self.gp.optimize('bfgs')

    def predict(self, X : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(X)
        
    def plot(self, X : np.ndarray, Y : np.ndarray, Y_pred : np.ndarray, sigma : np.ndarray):

        time = np.arange(X.shape[0])

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


        plt.subplot(3, 1, 2)
        plt.plot(time, Y)
        plt.errorbar(time, Y_pred, yerr=2*sigma, fmt='.r')
        plt.title('Prediction with error bar')
        plt.grid()


        # Plot error
        plt.subplot(3, 1, 3)
        plt.plot(time, Y-Y_pred, label='Output angle error')
        plt.xlabel('time')
        plt.ylabel('Estimation error (th)')
        plt.title('Sparse GP Estimation Error')
        plt.legend()
        plt.grid()

        plt.show()
        print ("Log-likelihood: {}".format(self.gp.log_likelihood()))
        