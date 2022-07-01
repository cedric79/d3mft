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
import time as pause
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class NN():
    """
    Class for training the neural network
    """
    
    def __init__(self,system_param, learn_param):
        """
        Initialise the neural network 
        """
        self.seedname = system_param["seedname"]
        self.n_samples = system_param["n_samples_train"]        
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
        print("Training the neural network\n")

        if self.train_grid_search:
            print("Doing a grid search")
            param_grid = dict(optimizer=self.gs_optimizers,
                              activation=self.gs_activations,
                              batch_size=self.gs_batch_sizes,
                              learn_rate=self.gs_learn_rates)
                
            perform_grid_search(beta=self.beta,
                                n_tau_target=self.n_tau_target, 
                                n_tau_in_data=self.n_tau_file,
                                regressor_epochs = self.regressor_epochs,
                                n_per_file=self.n_samples,
                                param_grid = param_grid,                                
                                preprocessing = "shift_and_rescale")
        if self.train_learning:
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
            print("-"*30, "> Saving Neural Network to " + self.seedname + ".h5 <" ,"-"*30)
            model.save(self.seedname+".h5")
            
        if self.train_prediction:
            model = tf.keras.models.load_model(self.seedname+".h5",
                                               custom_objects={'max_error': max_error, 'boundary_cond': boundary_cond})
            
            predict(model,
                    beta=self.beta,
                    n_tau_target=self.n_tau_target,
                    n_tau_in_data=self.n_tau_file,
                    n_per_file=self.n_samples,
                    train_files=self.train_files,
                    val_files=self.prediction_files,
                    samples=range(0,2))

def meta(beta):
    return "_{}".format(int(beta))

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
        # print(parent + prefix + meta + "_" + n + "_tau.csv")        
        data = np.loadtxt(parent + prefix + meta + "_" + n + "_tau.csv",
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
            if n < 10 :
                str_n = "0"+str(n)
            else:
                str_n = str(n)
                
            # X[row_start:row_middle, :n_tau] = get_data("G_PT", n)
            # X[row_start:row_middle, n_tau:] = get_data("G_SC", n)
            # Y[row_start:row_middle, :] = get_data("G_ED_Q", n)
            X[row_start:row_middle, :n_tau] = get_data("G_PT", str_n)
            X[row_start:row_middle, n_tau:] = get_data("G_SC", str_n)
            Y[row_start:row_middle, :] = get_data("G_ED_Q", str_n)
            
            # Data augmentation x2 - perform particle-hole transformation on gf
            # G(tau) = - G (beta - tau)
            # At strictly half filling this duplicates the database
            X[row_middle:row_end, :n_tau] = np.fliplr(X_train[row_start:row_middle, :n_tau])
            X[row_middle:row_end, n_tau:] = np.fliplr(X_train[row_start:row_middle, n_tau:])
            Y[row_middle:row_end, :] = np.fliplr(Y_train[row_start:row_middle, :])

    fill_with_data(X_train, Y_train, train_files)
    fill_with_data(X_test, Y_test, val_files)

    input_shape = (2 * n_tau, )
        
    return X_train, Y_train, X_test, Y_test, input_shape


def generate_data_tail(meta=meta(1), # temperature
                       n_tau=51, # col num of training
                       n_tau_in_data=201, # col num of inp
                       n_per_file=10, # number of entries per file 
                       train_files=[0, 1, 2, 3, 4, 5, 6, 7], # training set
                       val_files=[8, 9], # validation set
                       dtype='float16',
                       preprocessing="shift_and_rescale", # option of rescaling
                       parent="data/"):


    train_frac = 0.2
    n_train = n_per_file*train_frac 
    files_ = train_files + val_files  # concatenate all files  
    
    n_input = 2 * n_tau  # weak & strong coupling GF 
    n_output = n_tau  # approximation to an exact GF
    
    # skip maps n_tau_in
    skip = (n_tau_in_data - 1) // (n_tau - 1)


    def get_data_tail(prefix, n, maxr, skipped):
        # print(parent + prefix + meta + "_{}.csv".format(n))
        data = np.loadtxt(parent + prefix + meta + "_{}_tau.csv".format(n),
                          delimiter=",", max_rows = maxr, skip_rows = skipped)[:, ::skip]
        return transform(data, how=preprocessing)

    n_train = len(train_files) * n_per_file * 2
    n_test = len(val_files) * n_per_file * 2
    X_train = np.zeros((n_train, n_input), dtype=dtype)
    Y_train = np.zeros((n_train, n_output), dtype=dtype)
    X_test = np.zeros((n_test, n_input), dtype=dtype)
    Y_test = np.zeros((n_test, n_output), dtype=dtype)
    
    def fill_with_data_tail(X, Y, files, maxr, skipr):
        for i, n in enumerate(files):
            row_start = (2 * i) * n_in
            row_middle = (2 * i + 1) * n_in
            row_end = (2 * i + 2) * n_in
            X[row_start:row_middle, :n_tau] = get_data_tail("G_PT", n, maxr, skipr)
            X[row_start:row_middle, n_tau:] = get_data_tail("G_SC", n, )
            Y[row_start:row_middle, :] = get_data_tail("G_ED_Q", n)
            # Data augmentation x2 - perform particle-hole transformation on gf
            # G(tau) = - G (beta - tau)
            # At strictly half filling this duplicates the database
            X[row_middle:row_end, :n_tau] = np.fliplr(X_train[row_start:row_middle, :n_tau])
            X[row_middle:row_end, n_tau:] = np.fliplr(X_train[row_start:row_middle, n_tau:])
            Y[row_middle:row_end, :] = np.fliplr(Y_train[row_start:row_middle, :])


    nlines_train = n_train
    nlines_test  = n_per_file - n_train
    fill_with_data_tail(X_train, Y_train, files, n_train, 0, nlines_train)
    fill_with_data_tail(X_test, Y_test, files,n_per_file,n_train, nlines_test)

    input_shape = (2 * n_tau, )
        
    return X_train, Y_train, X_test, Y_test, input_shape


# Create NN
def max_error(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))

def boundary_cond(y_true, y_pred):
    preprocessing = 'shift_and_rescale'
    return K.max(K.abs(-(back_transform(y_pred[:, 0], how=preprocessing)
             + back_transform(y_pred[:, -1], how=preprocessing)) - 1.))

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

    model = Sequential()
    # input layer
    model.add(Dense(n_neurons,
                    input_shape=input_shape,
                    activation=activation,
                    kernel_initializer=init_mode))
    # hidden layers
    for i in range(n_hidden_layers - 1):
        model.add(Dense(n_neurons,
                        activation=activation,
                        kernel_initializer=init_mode))
    # output layer
    model.add(Dense(n_output, activation=activation, kernel_initializer=init_mode))

    # Create optimizer
    if optimizer == SGD:
        optimizer = SGD(lr=learn_rate, momentum=momentum)
    else:
        optimizer = optimizer(lr=learn_rate)

    loss = tensorflow.keras.losses.mean_squared_error
    # ES BUG: modify loss for legendre 
    metrics = ["mae", max_error, boundary_cond]
    
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_obj_prop(obj):
    for property, value in vars(obj).items():
        print(property, ":", value)

# Optimize hyperparameters of the model
def perform_grid_search(beta=1,
                        n_tau_target=51,
                        n_tau_in_data=201,
                        train_files=range(1,10),
                        val_files=[0],                        
                        regressor_epochs = 100,
                        n_per_file=10,
                        param_grid = "[]",
                        preprocessing='shift_and_rescale'):
    
    X_train, Y_train, X_test, Y_test, input_shape = generate_data(meta=meta(beta),
                                                                  n_tau=n_tau_target,
                                                                  n_tau_in_data=n_tau_in_data,
                                                                  n_per_file=n_per_file,
                                                                  preprocessing=preprocessing,
                                                                  train_files=train_files,
                                                                  val_files=val_files)

    model = KerasRegressor(build_fn=create_model,
                           epochs=regressor_epochs,
                           verbose=True)

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=-1)    
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    new_param = []
    for mean, stdev, param in zip(means, stds, params):
        param["mean"] = mean
        param["stdev"] = stdev
        new_param.append(param)

    # ES TODO: export new_param to pandas db
    # ES TODO: export pandas db to png 
    for i in new_param:
        print(i)

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
 
    
    model = create_model(preprocessing=preprocessing,
                         input_shape=input_shape,
                         n_output=n_tau_target,
                         activation=activation,
                         optimizer=optimizer,
                         learn_rate=learn_rate,
                         n_neurons=n_tau_target,
                         n_hidden_layers=n_hidden_layers)
    
    history = model.fit(X_train, Y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_data=(X_test, Y_test),
                        verbose=True)
    score = model.evaluate(X_test, Y_test, verbose=1)

    print("\n")
    print("*"*30, " Performance of Neural Network " ,"*"*30)
    print('Test MSE:', score[0])
    print('Test MAE:', score[1])
    print('Test max error:', score[2])
    print('Test boundary condition', score[3])
    print("*"*90)
            
    if plotting:
        subp = [2, 2, 1]
        metrics = ["loss", "mae", "max_error", "boundary_cond"]
        [ plotting_NN(history, i, subp) for i in metrics]                
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())        
        plt.show()
        print("\n")
        print("-"*30, "> Saving image: NN-details.png <" ,"-"*30)
        plt.savefig("NN-details.png")    
    
    return model

def plotting_NN(nn_hist, nn_metric, sub_plot):
    plt.subplot(*sub_plot); sub_plot[-1] += 1
    for metric in [nn_metric]:
        plt.plot(nn_hist.history[metric], color = "#121013")
        plt.plot(nn_hist.history['val_' + metric], color = "#eb596e")
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'val'], loc='best')

def predict(model,
            beta=1,
            n_tau_target=51,
            n_tau_in_data=201,
            n_per_file=10,
            train_files=[1,2],
            val_files=[0],
            preprocessing='shift_and_rescale',
            samples=range(3)):

    
    X_train, Y_train, X_test, Y_test, input_shape  = generate_data(meta=meta(beta),
                                                                   n_tau=n_tau_target,
                                                                   n_tau_in_data=n_tau_in_data,
                                                                   n_per_file=n_per_file,
                                                                   preprocessing=preprocessing,
                                                                   train_files=train_files, 
                                                                   val_files=val_files)
    
    Y = back_transform(model.predict(X_test[list(samples)]), how=preprocessing)
    tau = np.linspace(0, beta, n_tau_target, endpoint=True)
    
    for i, y in zip(samples, Y):
        plt.plot(tau, back_transform(Y_test[i], how=preprocessing), '-')
        plt.plot(tau, y, '--')
        plt.plot(tau, back_transform(X_test[i, :n_tau_target], how=preprocessing), '-.')
        plt.plot(tau, back_transform(X_test[i, n_tau_target:], how=preprocessing), '-.')
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$G(\tau)$")
        plt.legend(["ED_Q", "NN", "PT", "SC"], loc='best')
        plt.title("Sample {}".format(i))
        #mng = plt.get_current_fig_manager()
        #xmng.resize(*mng.window.maxsize())
        plt.show()
