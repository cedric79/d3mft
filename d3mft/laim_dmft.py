from triqs.gf import *
from triqs.dos import *
from triqs.plot.mpl_interface import oplot, plt
from triqs.operators import *
from h5 import *
import numpy as np
import random
import time
import os.path
import pandas as pd
from utility.mpi_tools import *
from utility.tools import *
from utility.db_gen import *
from solvers.PT_solver import *
from solvers.SC_solver import* 
import triqs.utility.mpi as mpi
from ml.nn import * 
#from triqs_cthyb import Solver
from read_config import * 
from scipy import interpolate
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Lambda, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow.compat.v1.keras.backend as K
import time as pause
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def solve_impurity(delta, solv_param, aim_param, solver):

    if solver == "PT":
        tau, GF_PT, gr = PT_aim_wrapper(delta, solv_param, aim_param)
        GF = GF_PT        
    if solver == "SC":
        tau, GF_SC, gr = SC_aim_wrapper(delta, solv_param, aim_param)
        GF = GF_SC
    if solver == "NN":
        #tau, GF = NN_aim_wrapper(delta, solv_param, aim_param)
        tau, GF, gr = NN_aim_wrapper(delta, solv_param, aim_param)
        # tau = []
        # GF = []
        # gr = []
        
    return tau, GF, gr

def NN_aim_wrapper(delta, solv_param, aim_param):
    model_name = "original_model.h5"
    model = tf.keras.models.load_model(model_name,
                                       custom_objects={'max_error': max_error, 'boundary_cond': boundary_cond})
    print(len(delta))
    tauPT, GF_PT = PT_aim_wrapper(delta, solv_param, aim_param)
    tauSC, GF_SC = SC_aim_wrapper(delta, solv_param, aim_param)
    gf = nn_gf(GF_PT[::200], GF_SC, model)
    tau, gf = interpolate_gf(tauPT[::200], gf, n_tau)
    #plt.plot(tauPT[::200], GF_PT[::200], 'o', label="PT")
    #plt.plot(tauSC[::4], GF_SC[::4], 's',label="SC")
    ##plt.plot(tauPT[::200], gf, label="NN")
    #plt.plot(tau, gf, '--',label="NN")
    gf_triqs = gf_to_triqs_new(tau, gf, "tau")
    g_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    g_iw << Fourier(gf_triqs)
    # analytical continuation 
    greal = GfReFreq(indices = [1], window = (-5.0,5.0))
    greal.set_from_pade(g_iw, 100, 0.01)                    
    # plt.legend()
    # plt.show()
    return tau, gf, greal

        
def PT_aim_wrapper(delta, solv_param, aim_param):
    """
    Solves the AIM using PT 
    """
    
    # initialise null parameters
    solv_param["file_to_write"] = "null"
    solv_param["file_to_write_iw"] = "null"
    # define delta
    tau = np.linspace(0, solv_param["beta"],  len(delta), endpoint=True)
    Delta_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau)
    Delta_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    # initialise the solver
    PT = PT_solver(solv_param)
    Delta_tau << gf_to_triqs_new(tau, delta, "tau")
    PT.Delta_iw << Fourier(Delta_tau)
    PT.G_0_iw << inverse(iOmega_n - param["eps"] - PT.Delta_iw)
    # solve
    PT.solve(param=param)
    # analytical continuation
    greal = GfReFreq(indices = [1], window = (-5.0,5.0))
    greal.set_from_pade(PT.G_iw, 100, 0.01)                
    tau, PT_gf=triqs_to_array(PT.G_tau,"tau")
    return tau, PT_gf#, greal

def SC_aim_wrapper(delta, solv_param, aim_param):
    # assumes delta comes in an array
    # make following 3 lines optional 
    # G_filenames = name("G_SC", beta, basis)
    # solv_param["file_to_write"] = G_filenames                
    # del_file(G_filenames)

    solv_param["file_to_write"] = "null"    
    tau = np.linspace(0, beta,  len(delta), endpoint=True)
    # ES BUG: remove hardcoded 50 here 
    solv_param["tau_arr"] = tau[::200]
    SC = SC_solver(solv_param)
    SC.solve(param=aim_param, delta_inp=delta[::200])
    greal = GfReFreq(indices = [1], window = (-5.0,5.0))
    greal.set_from_pade(SC.G_iw, 100, 0.01)         
    tau, SC_gf=triqs_to_array(SC.G_tr,"tau")    
    return tau, SC_gf#, greal

def DMFT(delta, solv_param, aim_param):
    print("Doing DMFT")
    n_iter = 25
    solver = "NN"
    alpha = 1.0
    delta_= []
    G_ =[]
    spec_ = []
    for iter in range(1,n_iter):
        print(iter, U, solv_param, aim_param)
        # solve AIM
        #tau, G = solve_impurity(delta, aim_params, solver)
        tau, G, gr = solve_impurity(delta, solv_param, aim_param, solver)
        # update delta
        #delta = update_delta(delta[::50], alpha, t, G)
        delta = update_delta(delta, alpha, t, G)
        # # get the spectral function        
        delta_.append(delta)
        G_.append(G)
        spec_.append(gr)

    # print("\n")
    # print(G_)
    # print("\n")
    for counter, i in enumerate(spec_):
        if counter % 5 == 0 : 
            oplot(-i.imag/np.pi)
    plt.show()
    # for counter, i in enumerate(delta_):
    #     plt.plot(tau, i, '-o',  label='iteration' +str(counter))
    #     #plt.plot(tau[::50], i[::50], '-o',  label='iteration' +str(counter))
    # plt.legend()
    # plt.show()
    
def update_delta(delta_in, alpha, t, G):
    new_delta = (1-alpha)*delta_in + alpha*(t**2)*G
    return new_delta
    

def run_DMFT(continuous_bath, beta, solv_param, aim_param):
    # Create initial hyrbididation!
    Delta_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    Delta_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau)
    Delta_iw <<  (t**2)*SemiCircular(2*t)
    Delta_tau << Fourier(Delta_iw)
    tau, delta = triqs_to_array(Delta_tau, "tau")

    DMFT(delta, solv_param, aim_param)

def triqs_to_array(dtau, basis):
    x,y = np.asarray(get_data(dtau, basis))
    return x,y

# Generating Delta
def Delta_from_func(Gamma_func, integrated_Gamma, E_max):
    d = DOSFromFunction(lambda x: Gamma_func(x) / integrated_Gamma,
                        -E_max, E_max, n_pts=1000)
    HT = HilbertTransform(d)
    Sigma0 = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    Sigma0.zero()
    return integrated_Gamma / np.pi * HT(Sigma=Sigma0)

# Semicircular functions 
def E_max(D):
    return D - 0.00001

def Gamma_func(D):
    def func(x):
        Gamma = D / 2
        return Gamma * np.sqrt(1 - (x / D)**2)
    return func

def integrated_Gamma(D):
    Gamma = D / 2
    return np.pi / 2 * D * Gamma

def interpolate_gf(tau, gf, new_n_tau):
    tck = interpolate.splrep(tau, gf, s=0.000001)
    new_tau = np.linspace(0, tau[-1], num=new_n_tau, endpoint=True)
    new_gf = interpolate.splev(new_tau, tck, der=0)
    return new_tau, new_gf

def load_nn_model(seedname):
    model = tf.keras.models.load_model(seedname+".h5")
    return model


def gf_to_triqs_new(x, y, basis):
    if basis == "matsu":
        print("work on below")
        #mesh_iw = MeshImFreq(beta, 'Fermion', n_max=1000)
        #gf=GfImFreq(indices=[0], beta=b_v, n_points = nomg, name=label)=        
        # print(len(gf.data))
        # ES BUG: this only populates the imaginary part now!!
        #gf.data[:,0,0] = 0.0+ 1j*g
    if basis == "tau": 
        gf = GfImTime(indices=[0], beta=beta, n_points=len(x))
        # ES BUG: only real part for the moment!
        gf.data[:,0,0] = y + 1j*0.0 #+ 1j*y_imag
    return gf

def nn_gf(weak_gf, strong_gf, model):
    preprocessing = "shift_and_rescale"
    n_tau = len(weak_gf) # get len
    print(n_tau)
    X = np.zeros((1, 2 * n_tau)) # create arr for input 
    X[0, :n_tau] = weak_gf #
    X[0, n_tau:] = strong_gf
    X = transform(X, preprocessing)
    Y = model.predict(X)
    Y = back_transform(Y, preprocessing)
    return Y[0]




if __name__ == "__main__":
    print("MAIN")
    # initialise delta
    #aim.G_iw << SemiCircular(2*t)
    #aim.Delta_iw = (t**2)*aim.G_iw
    #D = 4.  # half-bandwidth
    #t = D / 2.  # Bethe hopping
    Gamma = 1.  # impurity-bath coupling
    #beta = 1.  # inverse T
    indices = [0]  # only PM case considered
    n_iw = 1000
    n_tau = n_iw*10 + 1 # number of imaginary time points 

    # Local
    #U = 6. # Hubbard U
    #eps = 1.  # Local energy measured from eps_0 = -U/2 (eps = 0 <=> half-filling)
    N = 4  # number of bath sites

    D = 2.0
    t = 1.0
    U = 5.0
    beta = 1.0
    # n_loops = 2
    eps = 0.0
    
    n_iw = 1000 # number of matsubara points
    n_tau = n_iw*10 + 1 # number of imaginary time points
    n_l = 7 
    basis = "tau" # tau or legendre
    indices = [0]
    pt_order = "3" # order of the perturbation series
    target_n_tau = 200 # dimensionality of training input
    writing = False
    
    solv_param = {"beta": beta,
                  "n_iw": n_iw,
                  "n_l": n_l,
                  "writing": writing, 
                  "n_tau": n_tau,
                  "indices": indices,
                  "basis": basis,
                  "pt_order": pt_order,
                  "target_n_tau": target_n_tau}
    
    param = {"U": U,
             "eps":eps}
    
    continuous_bath=[Gamma_func(D), integrated_Gamma(D), E_max(D)]
    run_DMFT(continuous_bath, beta, solv_param, param)
