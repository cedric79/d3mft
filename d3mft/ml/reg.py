import os, time
import numpy as np
from math import tanh, atanh
from scipy import interpolate
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Lambda, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow.compat.v1.keras.backend as K
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
import sklearn.metrics as metrics
import time as pause
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))



class regress():
    """
    Class for doing regression
    """

    def __init__(self,system_param, learn_param):
        """
        Initialise the neural network 
        """
        self.seedname = system_param["seedname"]
        self.n_samples = system_param["n_samples"]
        self.beta = system_param["beta"]
        self.basis = system_param["basis"]
        self.n_tau_target = system_param["tau_learn"]
        self.n_tau_file = system_param["tau_file"]
        
        # grid search parameters         
        # ES TODO: figure out a way to un-hardcore the grid search
        self.regressor_epochs = 100
        self.gs_optimizers = [Nadam]
        self.gs_learn_rates = [0.0001, 0.0005, 0.001]
        self.gs_batch_sizes = [8]
        self.gs_activations = ['elu']
        # self.optimizers = [Adamax, Nadam]
        # learn_rates = [0.0001, 0.0005, 0.001]
        # batch_sizes = [8, 16, 32]
        # activations = ['elu', 'tanh']
        
        # ES BUG: if train_validation > (n_cores - train_validation) then error. Q: why?
        self.train_files = range(learn_param["train_validation"],
                                 system_param["n_cores"]) # ES BUG: make a choice in general, should it be system_param["n_cores"]-1 or not?
        self.val_files = range(0, learn_param["train_validation"])
        self.prediction_files = [0]
        self.train_activation = learn_param["train_activation"]
        self.train_learn_rate = learn_param["train_learn_rate"]
        self.train_n_neurons = learn_param["train_n_neurons"]
        self.train_n_hidden_layers = learn_param["train_n_hidden_layers"]
        self.train_batch_size = learn_param["train_batch_size"]
        self.train_epochs = learn_param["train_epochs"]
        self.train_plotting = learn_param["train_plotting"]
        self.train_grid_search = learn_param["train_grid_search"]
        self.train_learning = learn_param["train_learning"]
        self.train_prediction = learn_param["train_prediction"]

        if learn_param["train_optimizer"] == "Nadam":
            self.train_optimizer = Nadam
        elif learn_param["train_optimizer"] == "SGD":
            self.train_optimizer = SGD
        elif learn_param["train_optimizer"] == "RMSprop":
            self.train_optimizer = RMSprop
        elif learn_param["train_optimizer"] == "Adagrad":
            self.train_optimizer = Adagrad
        elif learn_param["train_optimizer"] == "Adadelta":
            self.train_optimizer = Adadelta
        elif learn_param["train_optimizer"] == "Adam":
            self.train_optimizer = Adam
        elif learn_param["train_optimizer"] == "Adamax":
            self.train_optimizer = Adamax
               

    def train(self):
        print("Training the regression model\n")
        print("Learning")
        model = perform_learning(beta = self.beta,
                                 n_tau_target = self.n_tau_target,
                                 n_tau_in_data = self.n_tau_file,
                                 train_files = self.train_files,
                                 val_files = self.val_files,                                     
                                 n_per_file = self.n_samples,
                                 activation = self.train_activation,
                                 optimizer = self.train_optimizer,
                                 learn_rate = self.train_learn_rate, 
                                 n_neurons = self.train_n_neurons,
                                 n_hidden_layers = self.train_n_hidden_layers,
                                 batch_size = self.train_batch_size,
                                 epochs = self.train_epochs,
                                 plotting = self.train_plotting)
            
def perform_learning(beta=1,
                     n_tau_target=51,
                     n_tau_in_data=201,
                     train_files=[2],
                     n_per_file=10,
                     val_files=[3],
                     activation="elu",
                     optimizer=Nadam,
                     learn_rate=0.0002,
                     n_neurons=51,
                     n_hidden_layers=2,
                     batch_size = 8,
                     epochs = 100,
                     plotting = True,
                     preprocessing='shift_and_rescale'):

            X_train, Y_train, X_test, Y_test, input_shape = generate_data(meta=meta(beta),
                                                                          n_tau=n_tau_target,
                                                                          n_tau_in_data=n_tau_in_data,
                                                                          n_per_file=n_per_file,
                                                                          preprocessing=preprocessing, 
                                                                          train_files=train_files,
                                                                          val_files=val_files)
            reg = LinearRegression().fit(X_train, Y_train)
            #reg = Ridge(alpha=0.5).fit(X_train, Y_train)
            #reg = Lasso(alpha=0.0001).fit(X_train, Y_train)
            #reg = BaesianRidge().fit(X_train, Y_train)
            prediction = reg.predict(X_test)[0]
            regression_results(prediction, Y_test[0])
            print(len(reg.coef_))
            print(reg.coef_)
            print(len(reg.coef_[0]))
            importance = reg.coef_
            # summarize feature importance
            for i,v in enumerate(importance):
                print('Feature:' ,i,  '  Score: ', v)
            #plt.bar([x for x in range(len(importance))], importance)
            #plt.show()
            # index_max = 4
            # for index in range(3,index_max):
            #     mod=reg.predict(X_test)[index]
            #     tau = np.linspace(0, beta, n_tau_target, endpoint=True)
            #     plt.plot(tau, X_test[index][:51], 'o',label="PT")
            #     plt.plot(tau, X_test[index][51:], 's',label="SC")
            #     plt.plot(tau, Y_test[index][:], '-',label="ED")
            #     plt.plot(tau, mod, '--',label="Regression")
            #     plt.legend()
            #     plt.title(str(index))
            #                plt.show()
            # print(reg.score(X_train,Y_train))
            # print(reg.coef_)
            # history = model.fit(X_train, Y_train)
            
            # history = model.fit(X_train, Y_train,
            #                     batch_size = batch_size,
            #                     epochs = epochs,
            #                     validation_data=(X_test, Y_test),
            #                     verbose=True)
            # score = model.evaluate(X_test, Y_test, verbose=1)

def create_model(preprocessing='shift_and_rescale',
                 input_shape=(2 * 51, ),
                 n_output=51,
                 activation='elu',
                 n_hidden_layers=2,
                 n_neurons=51,
                 optimizer=Nadam,
                 init_mode='uniform',
                 learn_rate=0.0002,
                 momentum=0.9,
                 kernel_size=5,
                 n_filters=8):

    print("create model")
    model = Lasso(alpha=1.0)
    
    # # input layer

    #loss = tensorflow.keras.losses.mean_squared_error
    #metrics = ["mae"]
    
    # # Compile model
    #model.compile(loss=loss, metrics=metrics)
    return model

def meta(beta):
    return "_{}".format(int(beta))


def generate_data(meta=meta(1), # temperature
                  n_tau=51, # col num of training
                  n_tau_in_data=201, # col num of inp
                  n_per_file=10, # number of entries per file 
                  train_files=[0, 1, 2, 3, 4, 5, 6, 7], # training set
                  val_files=[8, 9], # validation set
                  dtype='float16',
                  preprocessing="shift_and_rescale", # option of rescaling
                  parent="data/"):
    
    n_input = 2 * n_tau  # weak & strong coupling GF 
    n_output = n_tau  # approximation to an exact GF
    
    # skip maps n_tau_in
    skip = (n_tau_in_data - 1) // (n_tau - 1)


    def get_data(prefix, n):
        # print(parent + prefix + meta + "_{}.csv".format(n))
        data = np.loadtxt(parent + prefix + meta + "_{}_tau.csv".format(n),
                          delimiter=",")[:, ::skip]
        return transform(data, how=preprocessing)

    n_train = len(train_files) * n_per_file * 2
    n_test = len(val_files) * n_per_file * 2
    X_train = np.zeros((n_train, n_input), dtype=dtype)
    Y_train = np.zeros((n_train, n_output), dtype=dtype)
    X_test = np.zeros((n_test, n_input), dtype=dtype)
    Y_test = np.zeros((n_test, n_output), dtype=dtype)

    def fill_with_data(X, Y, files):
        for i, n in enumerate(files):
            row_start = (2 * i) * n_per_file
            row_middle = (2 * i + 1) * n_per_file
            row_end = (2 * i + 2) * n_per_file
            X[row_start:row_middle, :n_tau] = get_data("G_PT", n)
            X[row_start:row_middle, n_tau:] = get_data("G_SC", n)
            Y[row_start:row_middle, :] = get_data("G_ED_Q", n)
            # Data augmentation - perform particle-hole transformation on gf
            X[row_middle:row_end, :n_tau] = np.fliplr(X_train[row_start:row_middle, :n_tau])
            X[row_middle:row_end, n_tau:] = np.fliplr(X_train[row_start:row_middle, n_tau:])
            Y[row_middle:row_end, :] = np.fliplr(Y_train[row_start:row_middle, :])

    fill_with_data(X_train, Y_train, train_files)
    fill_with_data(X_test, Y_test, val_files)

    input_shape = (2 * n_tau, )
        
    return X_train, Y_train, X_test, Y_test, input_shape


def transform(data, how="shift_and_rescale"):
    if how == "minus":
        return -data
    if how == "shift":
        return data + 0.5
    if how == "shift_and_rescale":
        return (data + 0.5) * 2
    if how == "shift_and minus":
        return -(data + 0.5)
    if how == "shift_and_rescale_and_minus":
        return -(data + 0.5) * 2
    else:
        return data

def back_transform(data, how="shift_and_rescale"):
    if how == "minus":
        return -data
    if how == "shift":
        return data - 0.5
    if how == "shift_and_rescale":
        return 0.5 * data - 0.5
    if how == "shift_and minus":
        return -data - 0.5
    if how == "shift_and_rescale_and_minus":
        return -0.5 * data - 0.5
    else:        return data
