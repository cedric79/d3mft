import numpy as np
import time
from quspin.operators import hamiltonian, exp_op
from quspin.basis import spinful_fermion_basis_general
from triqs.gf import *
from triqs.dos import *
from triqs.operators import Operator, c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from triqs.plot.mpl_interface import oplot, plt
from laim.utility.db_gen import Delta_from_lists, save_gf, get_data
from laim.utility.mpi_tools import * 
from laim.utility.tools import plot_from_csv, get_obj_prop, name
from itertools import product

class ED_Q_solver():
    """
    Class for the ED pomerol solver 
    """

    def __init__(self, solver_params):
        """
        Initialise the PT_solver 
        """

        # general
        self.beta = solver_params["beta"]
        self.n_l = solver_params["n_l"]
        self.n_iw = solver_params["n_iw"]
        self.n_tau = solver_params["n_tau"]
        self.indices = solver_params["indices"]
        self.basis = solver_params["basis"]
        self.pt_order = solver_params["pt_order"]
        self.file_to_write = solver_params["file_to_write"]
        self.file_to_write_iw = solver_params["file_to_write_iw"]
        self.target_n_tau = solver_params["target_n_tau"]
        self.writing = solver_params["writing"]
        # self.energy_window = solver_params["energy_window"]
        # self.n_w = solver_params["n_w"]

        # Matsubara
        self.Delta_iw = GfImFreq(indices=self.indices,
                                 beta=self.beta,
                                 n_points=self.n_iw)        
        self.G_iw = self.Delta_iw.copy()
        self.G_0_iw = self.Delta_iw.copy()
        self.Sigma_iw = self.Delta_iw.copy()

        # Imaginary time
        self.G_tau_prev = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)
        self.G_tau = self.G_tau_prev.copy()
        self.G_tr = self.G_tau_prev.copy()        
        self.imtime_target = np.linspace(0, self.beta, 
                                         self.target_n_tau+1, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)        
        self.legendre_target = np.linspace(1, self.n_l, self.n_l, endpoint = True)

    def solve(self, param):
        tau=np.linspace(0., self.beta, self.n_tau, endpoint=True)
        self.e_list = param["e_list"]
        self.V_list = param["V_list"]
        self.eps = param["eps"]
        self.U = param["U"]
        G=calculate_G(self.beta, self.U, self.eps,
                      self.e_list, self.V_list, self.n_tau,
                      only_up_gf=True)
        self.G_tr=gf_to_triqs(G,self.beta,tau,"gtau")
        self.G_iw << Fourier(self.G_tr)
        self.G_l << MatsubaraToLegendre(self.G_iw)        
        #mesh_l = MeshLegendre(self.beta, 'Fermion', n_max=self.n_l)
        # G_l = GfLegendre(indices=self.indices,
        #                  mesh=mesh_l,
        #                  name=r'$G_l$')
        # G_iw_tmp = GfImFreq(indices=self.indices,
        #                     beta=self.beta,
        #                     n_points=self.n_iw)
                
    def write_data(self):
        """
        Writing database solution files for ED Q solver
        """

        only_gf=True
        tau_filename = name("G_ED_Q", self.beta, "tau")
        leg_filename = name("G_ED_Q", self.beta, "legendre")

        save_gf(tau_filename, self.G_tr, "tau",
                self.target_n_tau, only_gf=only_gf)                
        save_gf(leg_filename, self.G_l, "legendre",
                self.target_n_tau, only_gf=only_gf)
        
def calculate_G(beta, U, eps, e_list, V_list, n_tau, only_up_gf=False):
    spins = ["up", "down"]
    L = len(e_list) + 1
    bath_sites = range(1, L)
    N_list = list(range(L + 1))
    verbose = False
    def index(site, spin):
        return site + (0 if spin == "up" else L)

    def create_basis_and_hamiltonian(Nf, E0=0):
        basis = spinful_fermion_basis_general(L, simple_symm=False, Nf=Nf)
        # define site-coupling lists
        hyb_in_list = [[v, index(0, spin), index(i, spin)]
                       for v, i in zip(V_list, bath_sites) for spin in spins]
        hyb_out_list = [[-v, index(0, spin), index(i, spin)]
                        for v, i in zip(V_list, bath_sites) for spin in spins]
        pot_list = [[eps, index(0, spin)] for spin in spins]
        pot_list += [[e, index(i, spin)]
                     for e, i in zip(e_list, bath_sites) for spin in spins]
        int_list = [[U, index(0, "up"), index(0, "down")]]
        E0_list = [[-E0, 0]]
        # create static and dynamic lists for hamiltonian
        h_static = [
            ["+-", hyb_in_list],
            ["-+", hyb_out_list],
            ["n", pot_list],
            ["zz", int_list],
            ["I", E0_list],
        ]
        # create hamiltonian
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        H = hamiltonian(h_static, [], basis=basis, dtype=np.float64,
                        **no_checks)
        return basis, H

    def contribution_to_gf(N_up, N_down, gf_spin, E0=0):
        # Create basis
        Nf = [(N_up, N_down)] + ([(N_up + 1, N_down)] if gf_spin == "up"
                                 else [(N_up, N_down + 1)])
        basis, H = create_basis_and_hamiltonian(Nf, E0)
        if verbose:
            print("Sector (N_up={}, N_down={}). Extended basis size = {}".format(
            N_up, N_down, basis.Ns))
        # create operators
        cr = [["+", [[1, index(0, gf_spin)]]]]
        # an = [["-", [[1, index(0, gf_spin)]]]]
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        CR = hamiltonian(cr, [], basis=basis, dtype=np.float64, **no_checks)
        # AN = hamiltonian(an, [], basis=basis, dtype=np.float64, **no_checks)
        # Transform all operators to the eigenbasis
        E, W = H.eigh()
        CR = W.T @ CR.toarray() @ W
        AN = CR.T
        # Determine evolution operators in the eigenbasis
        tau = np.linspace(0, beta, n_tau, endpoint=True)
        U_tau = np.exp(np.outer(E, -tau))
        U_beta_minus_tau = np.fliplr(U_tau)
        # Calculate contributions to gf
        g = np.zeros(n_tau)
        if verbose: T_start = time()
        # g(tau) = -Tr[U(beta-tau) * AN * U(tau) * CR]
        g[:] = -np.einsum("ni,nm,mi,mn->i", U_beta_minus_tau, AN, U_tau, CR)
        if verbose:
            print("Calculation of {} GF in sector ({}, {}) completed in {:.2f} s".format(
            gf_spin, N_up, N_down, time() - T_start))
        return g

    # Find ground state energy
    E0_list = []
    for N_up, N_down in product(N_list, N_list):
        basis, H = create_basis_and_hamiltonian((N_up, N_down))
        E = H.eigvalsh()
        E0_list.append(min(E))
    E0 = min(E0_list)

    # Sum contributions to GF from all sectors
    G = np.zeros((2, n_tau))
    for g, gf_spin in zip(G, spins):
        if only_up_gf and gf_spin == "down": continue
        for N_up, N_down in product(N_list, N_list):
            if (N_up if gf_spin == "up" else N_down) == L: continue
            g[:] += contribution_to_gf(N_up, N_down, gf_spin, E0)
    # Normalize GF by demanding g(0) + g(beta) = -1
    Z = -(G[0, 0] + G[0, -1])
    G /= Z
    if only_up_gf:
        G[1] = G[0]

    return G

def gf_to_triqs(g,b_v,tau,label):
    #mesh_iw = MeshImFreq(beta, 'Fermion', n_max=1000)
    gf=GfImTime(indices=[0], beta=b_v, n_points = len(tau), name=label)
    gf.data[:,0,0] = g[0]
    #get_obj_prop(gf)
    return gf
