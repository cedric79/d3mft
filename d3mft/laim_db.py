import numpy as np
import random
import time
from laim.utility.mpi_tools import * 
from laim.utility.db_gen import *
from laim.utility.tools import *
from .read_config import *

def gen_db(system_param, db_param):
    """
    Main routine for the database generation
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
    indices = [0]
    
    # initialise different random seed on each core for each calculation
    rank = get_mpi_rank()
    seed = int((time.time()+ rank ))
    np.random.seed(seed)
    random.seed(seed)

    if rank == 0 : 
        print("\n#################  Generating database  #####################\n")
    
    if get_mpi_rank() == 0 :
        create_data_dir()
    mpi_barrier()
    
    filename=name_params("params", beta, basis)
    db_random = db_AIM(beta, U_, eps_, D_, V_, N_, samples_per_file, filename)
    db_random.create_db()
    db_random.pandas_db()
    mpi_barrier()
    
    if rank == 0:
        db_random.database.to_hdf('database.h5', key='df', mode='w')
        if database_distribution:
            print("------ Database viz -------")
            ax = db_random.database.hist(bins=50,
                                         xlabelsize=10, ylabelsize=10,
                                         figsize=(12,12))
            fig = ax[0][0].get_figure()
            fig.savefig("database_viz.png", format = "png")
            plt.show()
    mpi_barrier()
    
    if rank == 0 : 
        print("\n#################  Database generated  #####################\n")

    if rank == 0 : 
        print("\n#################  Generating hybridisation  #####################\n")

    Delta_filename = name("Delta", beta, basis)
    params = read_params(name_params("params", beta, basis))
    
    hyb_param = {"beta": beta,
                 "n_l": n_l,
                 "n_iw": n_iw,
                 "n_tau": n_tau,
                 "target_n_tau" : target_n_tau, 
                 "indices": indices,
                 "basis": basis,
                 "poly_semilog": poly_semilog}
    
    T_start = time.time()
    for p in params:
        export_Delta(hyb_param, bath_param=[p["e_list"], p["V_list"]], filename=Delta_filename, only_gf=True)        

    print("Generated {} Hybridisations in tau in {:.2f} s on core MPI core # {} ".format(len(params),time.time() - T_start, get_mpi_rank()))
    mpi_barrier()
    if rank == 0 : 
        print("\n#################  Hybridisation generated  #####################\n")
        if plot_hybrid_:
            # plot_param = {"sample_max": sample_max,
            #               "chosen_rank": chosen_rank}
            plot_param = {"sample_max": 4,
                          "chosen_rank": 0}            
            plot_param["file_to_read"] = Delta_filename
            plot_hybrid(plot_param, hyb_param)            
            
if __name__ == "__main__":

    input = "config.ini"
    system_param, db_param, aim_param, learn_param = get_config(input)
    gen_db(system_param, db_param)
