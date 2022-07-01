import numpy as np
import random
import time
import os.path
from subprocess import Popen
import pandas as pd
from laim.utility.db_gen import Delta_from_lists, save_gf, get_data
from laim.utility.mpi_tools import *
from laim.utility.tools import * 
from laim.solvers.PT_solver import *
#from laim.solvers.ED_POM_solver import *
from laim.solvers.ED_Q_solver import *
from laim.solvers.SC_solver import *
from .read_config import *
from laim.solvers.solv_funcs import * 

def solve_AIM(system_param, db_param, aim_param):
    """
    Main routine for solving AIM for the database
    """

    beta = system_param["beta"]
    U_ = db_param["U_"]
    eps_ = db_param["eps_"]
    D_ = db_param["D_"]
    V_ = db_param["V_"]
    N_ = db_param["N_"]
    database_distribution = db_param["database_distribution"]
    plot_hybrid_ = db_param["database_plot_hybrid"]
    samples_per_file = system_param["n_samples"]
    basis = system_param["basis"]
    n_l = db_param["n_l"]
    poly_semilog = db_param["poly_semilog"]
    n_iw = db_param["n_iw"]
    n_tau = db_param["n_tau"]
    target_n_tau = system_param["tau_file"]-1 # ES BUG: shouldn't have to -1 here, but crash if not
    writing = aim_param["aim_writing"]
    solvers = aim_param["solvers"]
    
    indices = [0] # ES TODO: should it be hardcoded?
    pt_order= "3" # ES TODO: should it be hardcoded?
    
    rank = get_mpi_rank()
    solv_param = {"beta": beta,
                  "n_l": n_l,
                  "n_iw": n_iw,
                  "n_tau": n_tau,
                  "indices": indices,
                  "basis": basis,
                  "pt_order": pt_order,
                  "target_n_tau": target_n_tau,
                  "writing": writing}
   
    # plot_param = {"sample_max": sample_max,
    #               "chosen_rank": chosen_rank}
    plot_param = {"sample_max": 4,
                  "chosen_rank": 0,
                  "basis": basis,
                  "poly_semilog": poly_semilog}
    
    params = read_params(name_params("params", beta, basis))
    
    if solvers["PT"]:
        if rank == 0:
            print("*"*20, " Running PT ", "*"*20)
            print("")
        G_filenames = name("G_PT", beta, basis)
        G_filenames_iw = name("G_PT", beta, "iw")
        solv_param["file_to_write"] = G_filenames
        solv_param["file_to_write_iw"] = G_filenames_iw
        delete_files("G_PT", beta)        
        for p in params:            
            PT = PT_solver(solv_param)
            PT.Delta_iw << Delta_from_lists(p["e_list"], p["V_list"])
            ## option 1: for PT DMFT
            # PT.G_iw << inverse(iOmega_n - p["eps"] - PT.Delta_iw - PT.Sigma_iw)
            # Insert DMFT loop here
            ## option 1: for PT DMFT
            # PT.G_0_iw << inverse(inverse(PT.G_iw) + PT.Sigma_iw)
            ## option 2: for standalone SIAM
            PT.G_0_iw << inverse(iOmega_n - p["eps"] - PT.Delta_iw)
            PT.solve(param=p)
            PT.write_data()

    if solvers["SC"]:
        if rank == 0:
            print("*"*20, " Running SC ", "*"*20)
            print("")
        G_filenames = name("G_SC", beta, basis)
        solv_param["file_to_write"] = G_filenames
        delete_files("G_SC", beta)
        # ES BUG: TODO absorb delta into params
        Delta_filename = name("Delta", beta, "tau")
        delta_db = pd.read_csv(Delta_filename, header = None)
        delta_arr = delta_db.to_numpy()
        tau = np.linspace(0, beta,  len(delta_arr[0]), endpoint=True)
        solv_param["tau_arr"] = tau
        for p,d in zip(params, delta_arr):            
            SC = SC_solver(solv_param)
            SC.solve(param=p, delta_inp=d)
            SC.write_data()
            
    if solvers["ED_Q"]: 
        if rank == 0:
            print("*"*20, " Running ED_Q", "*"*20)
            print("")
        G_filenames = name("G_ED_Q", beta, basis)
        G_filenames_iw = name("G_ED_Q", beta, "iw")
        solv_param["file_to_write"] = G_filenames
        solv_param["file_to_write_iw"] = G_filenames_iw
        #del_file(G_filenames)
        #del_file(G_filenames_iw)
        delete_files("G_ED_Q", beta)
        for p in params:
            ED_Q = ED_Q_solver(solv_param)
            ED_Q.solve(param=p)
            ED_Q.write_data()

    # split data if in serial mode
    if system_param["mpi_flag"] is False and rank ==0:
       cmd=system_param["code_loc"]+"split_data.sh "
       p = Popen(cmd,shell=True, executable='/bin/bash')
       p.wait()       
            
    comp_solutions = True
    if comp_solutions: 
        AIM=["PT", "ED_Q", "SC"]
        index = 1 # if we don't loop over indices
        comp(AIM, plot_param, index, "tau", beta, target_n_tau, n_l)
        comp(AIM, plot_param, index, "legendre", beta, target_n_tau, n_l)
        #sample_max = 1        
        #for index in range(0, sample_max):
        #comp(AIM, plot_param, index, "tau", beta, target_n_tau, n_l)

    # error_solutions = True
    # if error_solutions: 
    #     AIM=["PT", "ED_Q"]
    #     hist_rmsd(AIM, beta, basis, samples_per_file)
     
if __name__ == "__main__":

    input = "config.ini"
    system_param, db_param, aim_param, learn_param = get_config(input)
    solve_AIM(system_param, db_param, aim_param)
