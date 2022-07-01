import configparser
import json
import numpy as np

def get_config(config_file):
    # print("Reading config")
    config = configparser.ConfigParser()    
    config.read(config_file)

    # system parameters
    seedname = config.get("system", "seedname")
    code_loc = config.get("system", "root_dir")
    beta = config.getfloat("system", "beta")
    basis = config.get("system", "basis")
    tau_file = config.getint("system", "tau_file")
    tau_learn = config.getint("system", "tau_learn")
    n_cores = config.getint("system", "n_files")
    n_samples = config.getint("system", "n_samples_per_file")
    mpi_flag = config.getboolean("system", "mpi")
    
    if mpi_flag:
        n_samples_train = n_samples
    else:
        n_samples_train = n_samples
        n_samples = n_samples*n_cores
    
    system_param = {"seedname": seedname, 
                    "beta": beta,
                    "basis": basis,
                    "tau_file": tau_file,
                    "tau_learn": tau_learn,
                    "n_cores": n_cores,
                    "n_samples": n_samples,
                    "n_samples_train": n_samples_train, 
                    "mpi_flag": mpi_flag,
                    "code_loc": code_loc}

    # database parameters
    database_U = json.loads(config.get("database", "U_" ))
    database_eps = json.loads(config.get("database", "eps_" ))
    database_D = json.loads(config.get("database", "D_" ))
    database_V = json.loads(config.get("database", "V_" ))
    database_N = json.loads(config.get("database", "N_" ))
    n_iw = config.getint("database", "n_iw")
    n_l = config.getint("database", "n_l")
    poly_semilog = config.getboolean("database", "poly_semilog")
    database_distribution = config.getboolean("database", "database_distribution")
    database_plot_hybrid = config.getboolean("database", "plot_hybrid_")
    
    # conv string lists to float lists
    U_, eps_, D_, V_, N_=[], [], [], [], []
    [U_.append(float(i)) for i in database_U]
    [eps_.append(float(i)) for i in database_eps]
    [D_.append(float(i)) for i in database_D]
    [V_.append(float(i)) for i in database_V]
    [N_.append(float(i)) for i in database_N]

    # ES TODO: add more checks for different beta intervals
    if beta > 10.:
        assert n_l > 10, "Please specify n_l > 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " is too small"
        assert n_l < 40, "Please specify n_l < 40. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " will have too many zeros"
    elif beta < 5:
        assert n_l < 10, "Please specify n_l < 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " will have too many zeros"
    n_tau = n_iw*10 + 1 
    assert n_tau > n_iw*7, "Please make sure that you have many more n_tau than n_iw"

    db_param = {"U_": U_,
                "eps_": eps_,
                "D_": D_,
                "V_": V_,
                "N_": N_,
                "n_iw": n_iw,
                "n_tau": n_tau,
                "n_l": n_l,
                "database_plot_hybrid": database_plot_hybrid,
                "database_distribution": database_distribution,
                "poly_semilog": poly_semilog}

    # aim parameters
    aim_writing = config.getboolean("AIM", "writing")
    aim_solv = config.getboolean("AIM", "solv_on")
    aim_solvers = json.loads(config.get("AIM", "solvers" ))
    solvers = {}
    if aim_solv:
        for i in aim_solvers:
            solvers[i] = True
    else:
        for i in aim_solvers:
            solvers[i] = False
        
    aim_param = {"aim_writing": aim_writing,                 
                 "solvers": solvers}    
        
    # learning parameters
    train_optimizer = config.get("learn", "optimizer")
    train_activation = config.get("learn", "activation")
    train_learn_rate = config.getfloat("learn", "learn_rate")
    train_n_neurons = config.getint("learn", "n_neurons")
    train_n_hidden_layers = config.getint("learn", "n_hidden_layers")
    train_batch_size = config.getint("learn", "batch_size")
    train_epochs = config.getint("learn", "epochs")
    train_plotting = config.getboolean("learn", "plotting")
    train_val = config.getint("learn", "n_val")
    train_grid_search = config.getboolean("learn", "do_grid_search")
    train_learning = config.getboolean("learn", "do_learning")
    train_prediction = config.getboolean("learn", "do_prediction")
    
    learn_param = {"train_optimizer": train_optimizer, 
                   "train_activation": train_activation,
                   "train_learn_rate": train_learn_rate,
                   "train_n_neurons": train_n_neurons,
                   "train_n_hidden_layers": train_n_hidden_layers,
                   "train_batch_size": train_batch_size,
                   "train_epochs": train_epochs,
                   "train_plotting": train_plotting,
                   "train_validation": train_val,
                   "train_grid_search": train_grid_search,
                   "train_learning": train_learning,
                   "train_prediction": train_prediction}

    return system_param, db_param, aim_param, learn_param
