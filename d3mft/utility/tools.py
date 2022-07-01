import os
import os.path
import shutil
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from laim.utility.mpi_tools import * 

def name(prefix, beta, basis):
    """
    Naming function for the database files 
    """

    cwd = os.getcwd()
    parent = cwd + "/data/"
    rank = get_mpi_rank()
    if rank < 10:
        str_rank = "0"+str(rank)
    else:
        str_rank = str(rank)

        
    name_str = parent + prefix + "_" + str(int(beta)) \
               + "_" + str_rank + "_" + str(basis) + ".csv"

    return name_str 

def name_params(prefix, beta, basis):
    """
    Naming function for the database files 
    """

    cwd = os.getcwd()
    parent = cwd + "/data/"
    rank = get_mpi_rank()
    if rank < 10:
        str_rank = "0"+str(rank)
    else:
        str_rank = str(rank)
        
    name_str = parent + prefix + "_" + str(int(beta)) \
               + "_" + str_rank + ".csv"

    return name_str




def create_data_dir():
    """
    Creation/deletion of the data directory
    """
    
    print("Checking if ./data/ exists on root node")
    data_dir = os.getcwd()+"/data/"
    isdir = os.path.isdir(data_dir)
    if isdir: 
        print("Database already exists, we delete")
        shutil.rmtree(data_dir)
        print("Database is here: ", data_dir)
        os.mkdir(data_dir)    
    else:
        print("Database is here: ", data_dir)
        os.mkdir(data_dir)    

def read_params(filename):
    """
    Reads params from data/*.csv files 
    """
    
    params_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for line in reader:
            params = {}
            params["U"] = float(line[0])
            params["eps"] = float(line[1])
            N = int(float(line[3]))
            params["e_list"] = [float(e) for e in line[4:4+N]]
            params["V_list"] = [float(v) for v in line[4+N:4+2*N]]
            params_list.append(params)
    return params_list

def del_file(filename):     
    if os.path.isfile(filename):
        os.remove(filename)

def extract_from_csv(filename, index):
    y_axis = []
    with open(filename) as f:
        reader = csv.reader(f)
        interestingrows=[row for idx, row in enumerate(reader) if idx == index]
    [y_axis.append(float(i)) for i in interestingrows[0]]
    y_axis = np.asarray(y_axis)
    return y_axis
        
def plot_from_csv(filename, x_axis, index, descrip, axes, hyb_param):
    y_axis = extract_from_csv(filename, index)
    if hyb_param["basis"] == "legendre" and hyb_param["poly_semilog"]:        
        axes.semilogy(x_axis, np.abs(y_axis),
                      '-o', label = descrip)
        axes.set_ylim([1e-6,1e+1])
    else:
        axes.plot(x_axis, y_axis,
                  '-o', label = descrip)        
    return axes
    
    
def get_obj_prop(obj):        
    for property, value in vars(obj).items():
        print(property, ":", value)    
