import numpy as np
import csv
import random
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
from laim.utility.mpi_tools import * 
from triqs.gf import *
from triqs.dos import *
from laim.utility.tools import *

class db_AIM():
    """
    Class for the database object
    """

    def __init__(self, beta, U, eps, D, V, N, n_entries, filename):
        """
        Initialise the database. 

        Parameters
        ----------

        beta: scalar
        Inverse temparature

        U: list of length 2 of reals
        Range betwen U_min and U_max

        eps: list of length 2 of reals
        Range betwen eps_min and eps_max

        D: list of length 2 of reals
        Range betwen D_min and D_max (bandwidth)

        V: list of length 2 of reals
        Range betwen D_min and D_max (bath parameters)

        N: integer list 
        List of the number of bath parameters used 

        n_entries: integer
        number of entries per core in the database

        filename: string 
        name of the csv file for the database
        
        """
        
        self.beta = beta
        self.U = U
        self.eps = eps
        self.D = D
        self.V = V
        self.N = N
        self.n_entries = n_entries
        self.filename = filename
        self.data_entries = []
        
    def random_discrete_params(self):
        """
        Generate a set of randomly discrete parameters for the database
        """
 
        U = np.random.uniform(*self.U) # U 
        eps = np.random.uniform(*self.eps) # Filling 
        D = np.random.uniform(*self.D) # bandwidth 
        N = random.randint(*self.N) # num bath sites
        #print(U, eps, D, N)
        V_list = (self.V[1] - self.V[0]) * np.random.random(N) + self.V[0]
        V_list *=np.sqrt(2 * D / np.pi / sum(v**2 for v in V_list))
        #print(2 * D / pi, sum(v**2 for v in V_list))
        # print("V_i = ", V_list)
        e_list = np.sort(np.random.random(N))
        e_list -= sum(v**2 * e for v, e in zip(V_list, e_list)) / (2 * D / np.pi)
        e_list *= 2 * D / (e_list[-1] - e_list[0])
        # print("e_i = ", e_list)
        # print("")
        #print(0, sum(v**2 * e for v, e in zip(V_list, e_list)))
        #print(2 * D, e_list[-1] - e_list[0])
        params = [U, eps, D, N] + e_list.tolist() + V_list.tolist()
        self.data_entries.append(params)
        with open(self.filename, "ab") as f:
            np.savetxt(f, np.asarray(params)[:,np.newaxis].T, delimiter=",", fmt="%1.4f")
        
    def create_db(self):
        """
        create the database 
        """        
        [ self.random_discrete_params() for i in range(self.n_entries) ]            

    def pandas_db(self):
        """
        create a pandas dataframe with all entries of the database 
        and dump that into a database.txt file 
        """
        
        comm = get_mpi_comm()
        rank = get_mpi_rank()
        size = get_mpi_size()
        new_data = comm.gather( self.data_entries, root = 0)
        if rank == 0:
            N=random.randint(*self.N)
            header=["U", "eps", "D", "N_bath"]
            [ header.append("e_"+str(i)) for i in range(1,N+1) ]
            [ header.append("V_"+str(i)) for i in range(1,N+1) ]
            flatten = lambda t: [item for sublist in t for item in sublist]
            db = pd.DataFrame.from_records(flatten(new_data), columns = header)
            self.database = db
            assert db.shape[0] - self.n_entries*size < 1E-8, "Pandas database size != expected database size"
            # write into file
            fn = open("database.txt","w+")
            fn.write(db.to_string()+"\n")            
            fn.close()

def Delta_from_lists(e_list, V_list):
    """
    Returns hybridisation function
    """
    return sum(v**2 * inverse(iOmega_n - e) for e, v in zip(e_list, V_list))

def get_data(G, basis):
    mesh_=[]
    if basis == "iw": 
        for t in G.mesh:
            mesh_.append(t.value)
        #return [mesh_, G.data[:,0,0].real, G.data[:,0,0].imag]
        return [mesh_, G.data[:,0,0]]
    if basis == "tau":
        for t in G.mesh:
            mesh_.append(t.value)
        return [mesh_, G.data[:,0,0].real]
    if basis == "legendre":
        for t in G.mesh:
            mesh_.append(t.value)
        return [mesh_, G.data[:,0,0].real]
    
def save_gf(filename, G, basis, target_n_tau, only_gf=False):

    if basis == "iw": 
        data = np.asarray(get_data(G, basis))
        print(len(data[1]))
        print(data)
        target = 1
        skip_factor = 1
        #skip_factor=int((len(data[0])-1)/(target))
        if not only_gf:
            np.savetxt(filename, data.T[n_iw::skip_factor])
        else:
            with open(filename, 'ab') as f:
                # ES BUG: this only works for IM part at the moment
                # ES BUG: this needs to be changed to real and im part
                # ES BUG: perhaps just write a complex number
                #np.savetxt(f, data[2][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")
                np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")
                
    if basis == "tau": 
        data = np.asarray(get_data(G, basis))
        skip_factor=int((len(data[0])-1)/(target_n_tau))      
        if not only_gf:
            np.savetxt(filename, data.T[::skip_factor])
        else:
            with open(filename, 'ab') as f:
                np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")

    if basis == "legendre":
        data = np.asarray(get_data(G, basis))
        skip_factor=1
        if not only_gf:
            np.savetxt(filename, data.T[::skip_factor])
        else:
            with open(filename, 'ab') as f:
                np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")
                
def export_Delta(hyb_param, bath_param, filename="Delta.txt", only_gf=False):
    """
    Export Delta csv files per core in both legendre and tau bases
    """

    Delta_iw = GfImFreq(indices=hyb_param["indices"], beta=hyb_param["beta"],
                        n_points=hyb_param["n_iw"], name=r"$G(i \omega_n)$")
    
    e_list, V_list = bath_param
    Delta_iw << Delta_from_lists(e_list, V_list)
    
    mesh_l = MeshLegendre(beta=hyb_param["beta"], S = "Fermion",
                          n_max=hyb_param["n_l"])
    Delta_leg = GfLegendre(indices=hyb_param["indices"],
                           mesh=mesh_l, name=r'$G_l$')
    Delta_tau = GfImTime(indices=hyb_param["indices"], beta=hyb_param["beta"],
                         n_points=hyb_param["n_tau"], name=r"$\Delta(\tau)$")
    Delta_tau << Fourier(Delta_iw)        
    Delta_leg << MatsubaraToLegendre(Delta_iw)    
        
    only_gf = True
    tau_filename = name("Delta", hyb_param["beta"], "tau")
    leg_filename = name("Delta", hyb_param["beta"], "legendre")
    save_gf(tau_filename, Delta_tau, "tau", hyb_param["target_n_tau"], only_gf=only_gf)
    save_gf(leg_filename, Delta_leg, "legendre", hyb_param["target_n_tau"], only_gf=only_gf)
        
def plot_hybrid(plot_param, hyb_param):
    """
    Plot a sample of the hybridisation functions in the database
    """
    if hyb_param["basis"] == "tau": 
        x_axis = np.linspace(0, hyb_param["beta"], hyb_param["target_n_tau"]+1, endpoint=True)        
    if hyb_param["basis"] == "legendre":
        x_axis = np.linspace(1, hyb_param["n_l"], hyb_param["n_l"], endpoint = True)        
        
    rank = get_mpi_rank()
    if rank == plot_param["chosen_rank"]:
        fig,axes = plt.subplots(1)
        for index in range(0, plot_param["sample_max"]):
            label="Sample "+str(index)
            axes = plot_from_csv(plot_param["file_to_read"],
                                 x_axis,
                                 index,
                                 label,
                                 axes,
                                 hyb_param)
        axes.legend()
        if hyb_param["basis"] == "tau": 
            axes.set_xlabel(r"$\tau$")
            axes.set_ylabel(r"$\Delta(\tau)$")
            axes.set_title(r"Representative $\Delta(\tau)$ on core: " + str(rank) + "")
        if hyb_param["basis"] == "legendre":
            axes.set_xlabel(r"l")
            axes.set_ylabel(r"$\Delta_l$")
            axes.set_title(r"Representative $\Delta_l$ on core: " + str(rank) + "")
        plt.show()
