import numpy as np
import random
import time
import os.path
import pandas as pd
from laim.utility.db_gen import Delta_from_lists, save_gf, get_data
from laim.utility.mpi_tools import *
from laim.utility.tools import * 

def gf_to_triqs(g,b_v,nomg,label):
    #mesh_iw = MeshImFreq(beta, 'Fermion', n_max=1000)
    gf=GfImFreq(indices=[0], beta=b_v, n_points = nomg, name=label)
    # print(len(gf.data))
    # ES BUG: this only populates the imaginary part now!!
    gf.data[:,0,0] = 0.0+ 1j*g 
    # get_obj_prop(gf)
    return gf
           
def do_pade(gf,L,eta,wmin,wmax):
    
    g_pade = GfReFreq(indices = [0], window = (wmin,wmax))                       
    g_pade.set_from_pade(gf, n_points = L, freq_offset = eta)

    return g_pade

           
def hist_rmsd(AIM, beta, basis, samples_per_file):
    """
    RMSD plot of error relative to the true solution
    """
    
    comm = get_mpi_comm()
    rank = get_mpi_rank()
    size = get_mpi_size()
    y_ED_ = []
    y_PT_ = []
    deviations_ = []
    
    y_axis_PT = name("G_"+AIM[0], beta, basis)
    y_axis_ED = name("G_"+AIM[1], beta, basis)
    for i in range(0, samples_per_file):    
        y_ED_.append(extract_from_csv(y_axis_ED,i)) 
        y_PT_.append(extract_from_csv(y_axis_PT,i)) 

    log = False # ES BUG: maybe remove hardcode?
    if log: 
        for i,j in zip(y_ED_, y_PT_): 
            deviations_.append(np.log10(np.sum(np.abs(i - j)))/len(j))
    else:
        for i,j in zip(y_ED_, y_PT_): 
            deviations_.append(np.sum(np.abs(i - j))/len(j))
            
    all_deviations_ = comm.gather(deviations_, root = 0)
    if rank == 0:
        bins = 50
        flatten = lambda t: [item for sublist in t for item in sublist]
        all_deviations_ = flatten(all_deviations_)
        plt.hist(all_deviations_, bins) #, density=True, facecolor='g', alpha=0.75)
        plt.xlabel("RMSD error between "+ AIM[0]+ " and " +AIM[1]+ " solvers.")
        plt.ylabel("Number of occurances")
        plt.show()
        
        
def comp(AIM, plot_param, index, basis, beta, target_n_tau, n_l):
    """
    For PT solver plot a sample of the results
    """
    if basis == "tau": 
        x_axis = np.linspace(0, beta, target_n_tau+1, endpoint=True)        
    if basis == "legendre":
        x_axis = np.linspace(1, n_l, n_l, endpoint = True)        
    
    rank = get_mpi_rank()
    if rank == plot_param["chosen_rank"]:
        fig,axes = plt.subplots(1)
        for i in AIM:
            y_axis = name("G_"+i, beta, basis)
            print(y_axis)
            axes = plot_from_csv(y_axis, x_axis, index, i, axes, plot_param)
        axes.legend()
        axes.set_title("Comparison between solutions on rank = "+ str(rank))
        if basis == "tau": 
            axes.set_xlabel(r"$\tau$")
            axes.set_ylabel(r"$G(\tau)$")
        if basis == "legendre":
            axes.set_xlabel(r"l")
            axes.set_ylabel(r"$G_l$")        
        plt.show()

def delete_files(solver, beta):
    tau_filename = name(solver, beta, "tau")
    leg_filename = name(solver, beta, "legendre")
    del_file(tau_filename)
    del_file(leg_filename)


def analytical_cont(AIM):
    assert basis == "tau" , "Analytical continuation is not supported in the Legendre basis"
    x_axis = np.linspace(0, beta, target_n_tau+1, endpoint=True)

    wmin = -4.0
    wmax = 4.0
    eta = 0.01
    n_pade = 1001

    rank = get_mpi_rank()
    if rank == chosen_rank:
        for i in AIM:            
            y_name = name("G_"+i, beta, "iw")
            print("on rank=", rank, "and file is ", y_name )
            y_axis = extract_from_csv(y_name, 0)
            print(len(y_axis))
            # plt.plot(y_axis)
            # plt.show()
            g = gf_to_triqs(y_axis, beta, 1000, i)
            print(type(g))
            # oplot(g)
            # plt.show()
            a = do_pade(g, n_pade, eta, wmin, wmax)
            print(type(a))
            oplot(-a.imag/np.pi)
            plt.show()
            # print(g)
    #a = do_pade(g, n_pade, eta, wmin, wmax)
