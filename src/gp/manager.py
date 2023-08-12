# External libraries
import os
import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Local imports
from gp.gp import GaussianProcess, SparseGaussianProcess
from representation.narx import NARX
from config.definitions import DATASET_DIR, GP_MODELS_DIR, MODELS_DIR


class GPManager:
    """This class represents a manager for the Pendulum system.

    The manager is responsible for loading data, training the Gaussian Process (GP) model,
    testing the GP model, and plotting the results.
    """

    def __init__(self, num_inputs=3, num_outputs=3, sparse=True, num_inducing=1, num_samples=-1) -> None:
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
        self.save_model(model=self.pendulum_gp.gp,
                        nb=self.pendulum_gp.num_inputs,
                        na=self.pendulum_gp.num_outputs)
        
    def test(self, fname : str = '', load : bool = False, sim : bool = True):
        # load GP model
        if load:
            self.load_model(fname=fname)
        # test GP model
        print("Start testing...")
        if (sim):
            print("Start simulation...")
            X, Y = self.data["test"]
            Y_est, var = self.simulate(X=X)
            print("Y_sim shape: {}".format(Y_est.shape))
            print("var_sim shape: {}".format(var.shape))
            X_plot = X[self.io_max :]
            Y_plot = Y[self.io_max :]
        else:
            print("Start prediction...")
            X, Y = self.data_narx["test"]
            Y_est, var = self.predict(X=X)
            Y_est = Y_est.flatten()
            var = var.flatten()
            # print("Y_pred shape: {}".format(Y_est.shape))
            # print("var_pred shape: {}".format(var.shape))
            X_plot = X
            Y_plot = Y
        # plot data
        self.pendulum_gp.plot(X=X_plot, Y=Y_plot, Y_est=Y_est, var=var, sim=sim)
        return Y_est, var
    
    def test_prediction_submission(self, fname : str = ''):
        self.load_model(fname=fname)
        print(self.pendulum_gp.gp)

        file_name_data = os.path.join(DATASET_DIR, 'test-prediction-submission-file.npz')
        data = np.load(file_name_data)
        upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
        thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
        # thpred = data['thnow'] #all zeros

        # only select the ones that are used in the example
        Xtest = np.concatenate([upast_test[: , 15-self.num_inputs:], thpast_test[: , 15-self.num_outputs:]], axis=1)
        Ypredict, _ = self.predict(Xtest)
        assert len(Ypredict)==len(upast_test), 'number of samples changed!!'
        print(Ypredict)
        fname = "test-prediction-gp-submission-file"
        path_name = os.path.join(MODELS_DIR, fname + '.npz')
        with open(path_name, 'wb') as f:
            np.savez(f, upast=upast_test, thpast=thpast_test, thnow=Ypredict)
        print("Saving test prediction submission file in\n{}".format(path_name))

    def test_simulation_submission(self, fname : str = ''):
        self.load_model(fname=fname)
        print(self.pendulum_gp.gp)

        file_name_data = os.path.join(DATASET_DIR, 'test-simulation-submission-file.npz')
        data = np.load(file_name_data)
        u_test = data['u']
        th_test = data['th'] #only the first 50 values are filled the rest are zeros

        def simulation_IO_model(f, ulist, ylist, skip=50):
            upast = ulist[skip - self.num_inputs : skip].tolist() #good initialization
            ypast = ylist[skip - self.num_outputs : skip].tolist()
            
            Y = ylist[: skip].tolist()
            for u in ulist[skip: ]:
                x = np.concatenate([upast, ypast], axis=0)
                ypred = f(x)[0][0]
                Y.append(ypred)
                upast.append(u)
                upast.pop(0)
                ypast.append(ypred)
                ypast.pop(0)
            return np.array(Y)
        
        skip = 50
        th_test_sim = simulation_IO_model(lambda x: self.pendulum_gp.predict(x[None,:])[0], u_test, th_test, skip=skip)
        assert len(th_test_sim)==len(th_test)
        print("th_test_sim: {}".format(th_test_sim[0]))
        fname = "test-simulation-gp-submission-file"
        path_name = os.path.join(MODELS_DIR, fname + '.npz')
        with open(path_name, 'wb') as f:
            np.savez(f, th=th_test_sim, u=u_test)
        print("Saving test simulation submission file in\n{}".format(path_name))

    def grid_search(self):
        # na = np.arange(2, 9)
        # nb = np.arange(2, 9)
        # num_inducing = np.arange(20, 110, 20)

        na = [3]
        nb = [3]
        num_inducing = [100]

        nrms_dict = {}
        rms_dict = {}
        likelihood_dict = {}

        del self.pendulum_gp.kernel
        del self.pendulum_gp.gp
        del self.pendulum_gp
        del self.data
        del self.data_narx

        for _na in na:
            for _nb in nb:
                for _num_inducing in num_inducing:
                    print("\n---------------- NA: {}, NB: {}, NUM_INDUCING: {} ----------------".format(_na, _nb, _num_inducing))
                    # Instantiate data model representation
                    self.repr = NARX(num_inputs=_nb, num_outputs=_na)
                    # Load dataset
                    self.data, self.data_narx = self.load_data(
                        data_model=self.repr,
                        file_name=os.path.join(DATASET_DIR, 'training-data.csv'),
                        split_ratio=(0.7, 0.2, 0.1),    # 70% train, 20% validation, 10% test
                        num_samples=-1
                    )
                    X_train_narx, Y_train_narx = self.data_narx['train']
                    X_val_narx, Y_val_narx = self.data_narx['val']
                    
                    self.pendulum_gp = SparseGaussianProcess(X=X_train_narx,
                                        Y=Y_train_narx,
                                        num_inducing=_num_inducing,
                                        num_inputs=_nb,
                                        num_outputs=_na)
                    
                    self.train()
                    Y_est, var = self.test(sim=False)

                    print('train prediction errors:')
                    rms_rad = np.mean((Y_est-Y_val_narx)**2)**0.5
                    nrms = rms_rad/Y_val_narx.std()*100
                    print('RMS:', rms_rad,'radians')
                    print('RMS:', rms_rad/(2*np.pi)*360,'degrees')
                    print('NRMS:', nrms,'%')
                    print('Log-likelihood:', self.pendulum_gp.gp.log_likelihood()[0][0])
                    print("GP model", self.pendulum_gp.gp)
                    
                    nrms_dict[_nb, _na, _num_inducing] = nrms
                    rms_dict[_nb, _na, _num_inducing] = rms_rad
                    likelihood_dict[_nb, _na, _num_inducing] = self.pendulum_gp.gp.log_likelihood()[0][0]

                    del self.pendulum_gp.kernel
                    del self.pendulum_gp.gp
                    del self.pendulum_gp
                    del self.data
                    del self.data_narx

        max_rms_key = max(rms_dict)
        max_rms = max(rms_dict.values())
        min_rms_key = min(rms_dict)
        min_rms = min(rms_dict.values())
        print("Max RMS: key, value : {}, {}".format(max_rms_key, max_rms))
        print("Min RMS: key, value : {}, {}".format(min_rms_key, min_rms))

        max_nrms_key = max(nrms_dict)
        max_nrms = max(nrms_dict.values())
        min_nrms_key = min(nrms_dict)
        min_nrms = min(nrms_dict.values())
        print("Max NRMS: key, value : {}, {}".format(max_nrms_key, max_nrms))
        print("Min NRMS: key, value : {}, {}".format(min_nrms_key, min_nrms))

        max_likelihood_key = max(likelihood_dict)
        max_likelihood = max(likelihood_dict.values())
        min_likelihood_key = min(likelihood_dict)
        min_likelihood = min(likelihood_dict.values())
        print("Max LIKELIHOOD: key, value : {}, {}".format(max_likelihood_key, max_likelihood))
        print("Min LIKELIHOOD: key, value : {}, {}".format(min_likelihood_key, min_likelihood))

        path_name = os.path.join(GP_MODELS_DIR, f"gp_grid_search" + '.dump')
        with open(path_name, 'wb') as f:
            pickle.dump(nrms_dict, f, pickle.HIGHEST_PROTOCOL)
        
    def load_grid_search_dict(self):
        path_name = os.path.join(GP_MODELS_DIR, f"gp_grid_search" + '.dump')
        with open(path_name, 'rb') as f:
            print("Loading GP model...")
            nrms_dict = pickle.load(f)
        
        na = []
        nb = []
        inducing = []
        nrms = []
        for i, key in enumerate(nrms_dict):
            _nb, _na, _inducing = key
            nb.append(_nb)
            na.append(_na)
            inducing.append(_inducing)
            nrms.append(nrms_dict[key])
            # print("nb, na, inducing:{} - nrms: {}".format(key, nrms_dict[key]))
        print("NB", nb)
        print("NA", na)
        print("INDUCING", inducing)
        print("NRMS", nrms)

    def save_model(self, model, nb, na):
        print("Saving GP model...")
        fname = f"gp_induce{self.pendulum_gp.gp.num_inducing}_nb{nb}_na{na}"
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
            self.pendulum_gp.num_inputs = self.num_inputs
            self.pendulum_gp.num_outputs = self.num_outputs
            self.num_inducing = self.pendulum_gp.gp.num_inducing
            print("Loading GP kernel : {}".format(self.pendulum_gp.kernel))
            print("Loading GP {} inducing points Z".format(self.pendulum_gp.gp.num_inducing))

    def load_data(self, data_model, file_name : str, split_ratio : tuple = (0.7, 0.2, 0.1), num_samples: int = -1, verbose : bool = 0) -> dict:
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
        
        if verbose == 1:
            # np.set_printoptions(threshold=sys.maxsize)
            print("\n------------ GENERAL INFO ------------")
            print("Number of inputs, outputs: {}, {}".format(self.num_inputs, self.num_outputs))
            print("Data input type: {}, with shape: {}".format(type(X), X.shape))
            print("Data output type: {}, with shape: {}".format(type(Y), Y.shape))

            print("\n------------ VANILLA DATASET MANIPULATION/SPLITTING ------------")

        len_data = X.shape[0]
        # load subset of data
        train_samples = int(split_ratio[0] * len_data)     # 0.7 * (800000 - max(na,nb)) = 56000
        val_samples = int(split_ratio[1] * len_data)       # 0.2 * (800000 - max(na,nb)) = 16000
        test_samples = int(split_ratio[2] * len_data)      # 0.1 * (800000 - max(na,nb)) = 8000
        if verbose == 1:
            print("train samples : {}".format(train_samples))
            print("validation samples : {}".format(val_samples))
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
        
        if verbose == 1:
            print("Train/Val/Test inputs shapes : {}/{}/{}".format(X_train.shape, X_val.shape, X_test.shape))
            print("Train/Val/Test outputs shapes : {}/{}/{}".format(Y_train.shape, Y_val.shape, Y_test.shape))
            print('Dataset loaded with %d training samples, %d validation samples, %d test samples!' % (train_samples, val_samples, test_samples))

            print("\n------------ PARSE DATASET THROUGH NARX ------------")
        # Pass data through model representation
        X_narx, Y_narx = data_model.make_training_data(X, Y)
        if verbose == 1:
            print("Features type: {}, with shape: {}".format(type(X_narx), X_narx.shape))
            print("Outputs type: {}, with shape: {}".format(type(Y_narx), Y_narx.shape))

            print("\n------------ NARX DATASET MANIPULATION/SPLITTING ------------")
        len_data_narx = X_narx.shape[0]
        # load subset of data
        train_narx_samples = int(split_ratio[0] * len_data_narx)     # 0.7 * (800000 - max(na,nb)) = 56000
        val_narx_samples = int(split_ratio[1] * len_data_narx)       # 0.2 * (800000 - max(na,nb)) = 16000
        test_narx_samples = int(split_ratio[2] * len_data_narx)      # 0.1 * (800000 - max(na,nb)) = 8000
        if verbose == 1:
            print("train narx samples : {}".format(train_narx_samples))
            print("validation narx samples : {}".format(val_narx_samples))
            print("test narx samples : {}".format(test_narx_samples))

        X_train_narx = X_narx[:train_narx_samples]
        Y_train_narx = Y_narx[:train_narx_samples]

        X_val_narx = X_narx[train_narx_samples : train_narx_samples+val_narx_samples]
        Y_val_narx = Y_narx[train_narx_samples : train_narx_samples+val_narx_samples]

        X_test_narx = X_narx[val_narx_samples : val_narx_samples+test_narx_samples]
        Y_test_narx = Y_narx[val_narx_samples : val_narx_samples+test_narx_samples]

        if verbose == 1:
            print("NARX: Train/Val/Test features shapes : {}/{}/{}".format(X_train_narx.shape, X_val_narx.shape, X_test_narx.shape))
            print("NARX: Train/Val/Test outputs shapes : {}/{}/{}".format(Y_train_narx.shape, Y_val_narx.shape, Y_test_narx.shape))
            print('Dataset NARX loaded with %d training samples, %d validation samples, %d test samples!' % (train_narx_samples, val_narx_samples, test_narx_samples))
            print()

        # return data
        data_narx = {
            'train' : (X_train_narx, Y_train_narx),
            'val'   : (X_val_narx, Y_val_narx),
            'test'  : (X_test_narx, Y_test_narx)
        }
        
        return data, data_narx
