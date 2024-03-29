# External libraries
import torch
import numpy as np
from typing import Callable
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

    def plot(self, X : np.ndarray, Y : np.ndarray, Y_est : np.ndarray, var : np.ndarray) -> None:
        """
        Plot the original data, predictions, and confidence intervals.

        Parameters:
        X (numpy.ndarray): Input variables.
        Y (numpy.ndarray): Original output variables.
        Y_est (numpy.ndarray): Predicted output variables.
        var (numpy.ndarray): Standard deviations of the predictions.
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
        plt.errorbar(time, Y_est, yerr=2*var, fmt='.r')
        plt.xlabel('time')
        plt.ylabel('Output angle (th)')
        plt.title('GP Prediction with error')
        plt.legend()
        plt.grid()

        # Plot error
        plt.subplot(3, 1, 3)
        plt.plot(time, Y-Y_est, label='Output angle error')
        plt.xlabel('time')
        plt.ylabel('Estimation error (th)')
        plt.title('GP Estimation Error')
        plt.legend()
        plt.grid()

        plt.show()


class SparseGaussianProcess:
    """This class represents a Sparse Gaussian Process (SGP) model for a Pendulum system."""
    def __init__(self, X : np.ndarray, Y : np.ndarray, num_inducing : int, num_inputs : int, num_outputs : int) -> None:
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.io_max = max(num_inputs, num_outputs)
        # define kernel
        self.kernel = GPy.kern.RBF(input_dim=num_inputs+num_outputs, lengthscale=1.0, variance=1.0)
        
        min_x, max_x = np.min(X), np.max(X)
        inducing_points = np.random.uniform(low=min_x, high=max_x, size=(num_inducing, num_inputs + num_outputs))
        # define regressor
        if Y.ndim == 1:
            Y = Y[:, None]
        self.gp = GPy.models.SparseGPRegression(X, Y, kernel=self.kernel, Z=inducing_points)
        self.Z = self.gp.Z
        print('%s initialized with kernel:%s' % (__class__.__name__, self.kernel))

    def fit(self, X : np.ndarray, Y : np.ndarray) -> None:
        self.gp.optimize('bfgs')

    def predict(self, X : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(X)
    
    def simulate(self, X : np.ndarray, repr) -> tuple[np.ndarray, np.ndarray]:
        # sim = repr.simulate(X_test[:, 0], f=self.pendulum_gp.predict)
        sim = repr.simulate(X, f=self.predict)
        return sim['mean'][self.io_max :], sim['var'][self.io_max :]

    def plot(self, X : np.ndarray, Y : np.ndarray, Y_est : np.ndarray, var : np.ndarray, sim : bool):

        time = np.arange(X.shape[0])

        print('Stats:')
        rms_rad = np.mean((Y_est-Y)**2)**0.5
        rms_deg = rms_rad/(2*np.pi)*360
        nrms = rms_rad/Y.std()*100
        print('RMS:', rms_rad,'radians')
        print('RMS:', rms_deg,'degrees')
        print('NRMS:', nrms,'%')

        # Plot original data
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.plot(time, X, label='Input Voltage u (GT)')
        # plt.plot(time, Y, label='Output Angle th (GT)')
        # # plt.xlabel('time')
        # plt.ylabel('Input Voltage (u)')
        # plt.title('Original Pendulum Data')
        # plt.legend()
        # plt.grid()
        # time = time[0:800]
        # Y = Y[0:800]
        # Y_est = Y_est[0:800]
        # var = var[0:800]

        plt.subplot(2, 1, 1)
        plt.plot(time, Y, label='Ground truth angle')
        plt.plot(time, Y_est)
        # plt.errorbar(time, Y_est, yerr=2*var, fmt='.r')
        plt.errorbar(time, Y_est, yerr=2*var, fmt='.r', label='Estimated angle')
        if sim:
            plt.title('Sparse GP Simulation with Error bar')
        else:
            plt.title('Sparse GP Prediction with Error bar')
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        plt.grid()

        # Plot error
        plt.subplot(2, 1, 2)
        plt.plot(time, Y, label='Ground truth angle')
        plt.plot(time, Y-Y_est, label='Residuals angle error')
        plt.xlabel('time')
        plt.ylabel('Theta angle')
        if sim:
            plt.title('Sparse GP Simulation Error')
        else:
            plt.title('Sparse GP Prediction Error')
        
        plt.legend()
        plt.grid()

        plt.show()
        print ("Log-likelihood: {}".format(self.gp.log_likelihood()))
        