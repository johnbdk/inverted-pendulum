# External libraries
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Local imports
from gp.gp import GaussianProcess, SparseGaussianProcess
from representation.narx import NARX
from config.definitions import DATASET_DIR, GP_MODELS_DIR


class GPManager:
    """This class represents a manager for the Pendulum system.

    The manager is responsible for loading data, training the Gaussian Process (GP) model,
    testing the GP model, and plotting the results.
    """

    def __init__(self, num_inputs=3, num_outputs=3, sparse=True, num_inducing=0, num_samples=-1) -> None:
        """
        Constructor for GPManager.

        Upon initialization, the manager loads the data, and initializes the choosen GP model.

        Attributes:
        num_inputs: Number of past inputs to be used (nb)
        num_outputs: Number of past inputs to be used (na)
        sparse: Flag indicating sparse GP or not
        num_inducing: Number of inducing points (used in sparse GP)
        io_max: Number indicating the maximum past info between nb and na
        repr: Data model representation (eg. NARX)
        pendulum_gp (GaussianProcess or SparseGaussianProcess): The GP model for the pendulum system.
        """

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.io_max = max(self.num_inputs, self.num_outputs)
        
        # Instantiate data model representation
        self.repr = NARX(num_inputs=self.num_inputs, num_outputs=self.num_outputs)

        # Load dataset
        self.data, self.data_narx = self.load_data(
            data_model=self.repr,
            file_name=os.path.join(DATASET_DIR, 'training-data.csv'),
            split_ratio=(0.7, 0.2, 0.1),    # 70% train, 20% validation, 10% test
            num_samples=num_samples
        )

        # define the gaussian process model
        if sparse:
            X_train_narx, Y_train_narx = self.data_narx['train']
            self.pendulum_gp = SparseGaussianProcess(X=X_train_narx,
                                                     Y=Y_train_narx,
                                                     num_inducing=self.num_inducing,
                                                     num_inputs=self.num_inputs,
                                                     num_outputs=self.num_outputs)
        else:
            self.pendulum_gp = GaussianProcess()

    def simulate(self, X : np.ndarray):
        Y_sim, var_sim = self.pendulum_gp.simulate(X=X, repr=self.repr)
        return Y_sim, var_sim
    
    def predict(self, X : np.ndarray):
        Y_pred, var_pred = self.pendulum_gp.predict(X=X)
        return Y_pred, var_pred
    
    def train(self):
        print("Start training...")
        # train GP model
        X, Y = self.data_narx["train"]
        self.pendulum_gp.fit(X=X, Y=Y)
        self.save_model(model=self.pendulum_gp.gp)
        
    def test(self, fname : str = '', sim : bool = True):
        # load GP model
        self.load_model(fname=fname)
        # test GP model
        print("Start testing...")
        X, Y = self.data["test"]
        if (sim):
            print("Start simulation...")
            Y_est, var = self.simulate(X=X)
            print("Y_sim shape: {}".format(Y_est.shape))
            print("var_sim shape: {}".format(var.shape))
        else:
            print("Start prediction...")
            Y_est, var = self.predict(X=X)
            print("Y_pred shape: {}".format(Y_est.shape))
            print("var_pred shape: {}".format(var.shape))
        # plot data
        self.pendulum_gp.plot(X=X[self.io_max :], Y=Y[self.io_max :], Y_est=Y_est, var=var)
    
    def grid_search(self):
        pass

    def save_model(self, model):
        print("Saving GP model...")
        fname = f"gp_induce{self.pendulum_gp.gp.num_inducing}_nb{self.num_inputs}_na{self.num_outputs}"
        path_name = os.path.join(GP_MODELS_DIR, fname + '.dump')
        with open(path_name, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print("Saving GP kernel : {}".format(self.pendulum_gp.kernel))
        print("Saving GP {} inducing points Z".format(self.pendulum_gp.gp.num_inducing))

    def load_model(self, fname):
        assert fname != ''
        assert fname != None
        path_name = os.path.join(GP_MODELS_DIR, fname + '.dump')
        print("User's requested path: {}".format(path_name))
        with open(path_name, 'rb') as f:
            print("Loading GP model...")
            self.pendulum_gp.gp = pickle.load(f)
            self.pendulum_gp.kernel = self.pendulum_gp.gp.kern
            self.pendulum_gp.io_max = self.io_max
            self.pendulum_gp.Z = self.pendulum_gp.gp.Z
            self.num_inducing = self.pendulum_gp.gp.num_inducing
            print("Loading GP kernel : {}".format(self.pendulum_gp.kernel))
            print("Loading GP {} inducing points Z".format(self.pendulum_gp.gp.num_inducing))

    def load_data(self, data_model, file_name : str, split_ratio : tuple = (0.7, 0.2, 0.1), num_samples: int = -1) -> dict:
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
        if (num_samples == -1):
            data = pd.read_csv(file_name)
        else:
            data = pd.read_csv(file_name)[:num_samples]
        X = data['u'].values
        Y = data['th'].values

        # np.set_printoptions(threshold=sys.maxsize)
        print("\n------------ GENERAL INFO ------------")
        print("Number of inputs, outputs: {}, {}".format(self.num_inputs, self.num_outputs))
        print("Data input type: {}, with shape: {}".format(type(X), X.shape))
        print("Data output type: {}, with shape: {}".format(type(Y), Y.shape))

        print("\n------------ VANILLA DATASET MANIPULATION/SPLITTING ------------")
        len_data = X.shape[0]
        # load subset of data
        train_samples = int(split_ratio[0] * len_data)     # 0.7 * (800000 - max(na,nb)) = 56000
        print("train samples : {}".format(train_samples))
        val_samples = int(split_ratio[1] * len_data)       # 0.2 * (800000 - max(na,nb)) = 16000
        print("validation samples : {}".format(val_samples))
        test_samples = int(split_ratio[2] * len_data)      # 0.1 * (800000 - max(na,nb)) = 8000
        print("test samples : {}".format(test_samples))
        
        X_train = X[:train_samples]
        Y_train = Y[:train_samples]

        X_val = X[train_samples : train_samples+val_samples]
        Y_val = Y[train_samples : train_samples+val_samples]

        X_test = X[val_samples : val_samples+test_samples]
        Y_test = Y[val_samples : val_samples+test_samples]

        data = {
            'train' : (X_train, Y_train),
            'val'   : (X_val, Y_val),
            'test'  : (X_test, Y_test)
        }
        
        print("Train/Val/Test inputs shapes : {}/{}/{}".format(X_train.shape, X_val.shape, X_test.shape))
        print("Train/Val/Test outputs shapes : {}/{}/{}".format(Y_train.shape, Y_val.shape, Y_test.shape))
        print('Dataset loaded with %d training samples, %d validation samples, %d test samples!' % (train_samples, val_samples, test_samples))

        print("\n------------ PARSE DATASET THROUGH NARX ------------")
        # Pass data through model representation
        X_narx, Y_narx = data_model.make_training_data(X, Y)
        print("Features type: {}, with shape: {}".format(type(X_narx), X_narx.shape))
        print("Outputs type: {}, with shape: {}".format(type(Y_narx), Y_narx.shape))

        print("\n------------ NARX DATASET MANIPULATION/SPLITTING ------------")
        len_data_narx = X_narx.shape[0]
        # load subset of data
        train_narx_samples = int(split_ratio[0] * len_data_narx)     # 0.7 * (800000 - max(na,nb)) = 56000
        print("train narx samples : {}".format(train_narx_samples))
        val_narx_samples = int(split_ratio[1] * len_data_narx)       # 0.2 * (800000 - max(na,nb)) = 16000
        print("validation narx samples : {}".format(val_narx_samples))
        test_narx_samples = int(split_ratio[2] * len_data_narx)      # 0.1 * (800000 - max(na,nb)) = 8000
        print("test narx samples : {}".format(test_narx_samples))

        X_train_narx = X_narx[:train_narx_samples]
        Y_train_narx = Y_narx[:train_narx_samples]

        X_val_narx = X_narx[train_narx_samples : train_narx_samples+val_narx_samples]
        Y_val_narx = Y_narx[train_narx_samples : train_narx_samples+val_narx_samples]

        X_test_narx = X_narx[val_narx_samples : val_narx_samples+test_narx_samples]
        Y_test_narx = Y_narx[val_narx_samples : val_narx_samples+test_narx_samples]

        print("NARX: Train/Val/Test features shapes : {}/{}/{}".format(X_train_narx.shape, X_val_narx.shape, X_test_narx.shape))
        print("NARX: Train/Val/Test outputs shapes : {}/{}/{}".format(Y_train_narx.shape, Y_val_narx.shape, Y_test_narx.shape))
        print('Dataset NARX loaded with %d training samples, %d validation samples, %d test samples!' % (train_narx_samples, val_narx_samples, test_narx_samples))
        print()
        
        # return data
        narx_data = {
            'train' : (X_train_narx, Y_train_narx),
            'val'   : (X_val_narx, Y_val_narx),
            'test'  : (X_test_narx, Y_test_narx)
        }
        
        return data, narx_data
