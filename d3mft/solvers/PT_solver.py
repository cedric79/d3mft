import numpy as np
import time
from triqs.gf import *
from triqs.dos import *
from triqs.plot.mpl_interface import oplot, plt
from laim.utility.db_gen import Delta_from_lists, save_gf, get_data
from laim.utility.mpi_tools import * 
from laim.utility.tools import plot_from_csv, name, del_file

class PT_solver():
    """
    Class for a perturbation theory solver 
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
        
        # Matsubara
        self.Delta_iw = GfImFreq(indices=self.indices,
                                 beta=self.beta,
                                 n_points=self.n_iw)
        self.G_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.G_0_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.Sigma_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)

        # Imaginary time
        self.G_tau_prev = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)
        self.G_tau = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)
        self.imtime_target = np.linspace(0, self.beta, 
                                         self.target_n_tau+1, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)        
        self.legendre_target =  np.linspace(1, self.n_l, self.n_l, endpoint = True)
        
    def solve(self, param):
        """
        Solve the AIM using PT
        """        
        self.eps = param["eps"]
        self.U = param["U"]
        self.G_tau_prev <<  Fourier(self.G_0_iw)
        self.solv_AIM()
        self.G_l << MatsubaraToLegendre(self.G_iw)
        
    def write_data(self):
        """
        Writing database solution files for PT solver
        """
        only_gf = True            
        tau_filename = name("G_PT", self.beta, "tau")
        leg_filename = name("G_PT", self.beta, "legendre")
        save_gf(leg_filename, self.G_l, "legendre",
                self.target_n_tau, only_gf=only_gf)
        save_gf(tau_filename, self.G_tau, "tau",
                self.target_n_tau, only_gf=only_gf)                
        
    def solv_AIM(self):
        """
        PT solver for the AIM
        """
        
        self_consistent = False
        tol, n_iter_max = 1e-5, 1
        for iter in range(n_iter_max):
            
            if self.pt_order == "1":
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U))
                              
            if self.pt_order == "2":
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U) \
                                  + second_order_Sigma(self.G_tau_prev, self.U, only_skeleton=self_consistent))
                
            if self.pt_order == "3": 
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U) \
                                  + second_order_Sigma(self.G_tau_prev, self.U, only_skeleton=self_consistent) \
                                  + third_order_Sigma(self.G_tau_prev, self.U, self.indices, 
                                                      self.n_iw, only_skeleton=self_consistent))
            
            self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw)
            self.G_tau << Fourier(self.G_iw)
            
            if np.allclose(self.G_tau_prev.data, self.G_tau.data, atol=tol) or not self_consistent:
                # print("Converged in iteration {}".format(iter))
                return self.G_tau
            else:
                self.G_tau_prev << 0.8 * self.G_tau + 0.2 * self.G_tau_prev
                # print("Solution not converged!")
        return self.G_tau
    
def reverse_tau(G_tau, statistic="Fermion"):
    sign = -1 if statistic == "Fermion" else 1
    G_minus_tau = G_tau.copy()
    G_minus_tau.data[:,0,0] = sign * np.flipud(G_tau.data[:,0,0])
    # ES BUG: TODO add tail with TRIQS V3
    # for m in range(G_tau.tail.order_min, G_tau.tail.order_max + 1)    
    #     G_minus_tau.tail[m] = (-1)**m * G_tau.tail[m]
    return G_minus_tau

def trapez(X, dtau):
    if len(X) < 2: return 0
    I = dtau * np.sum(X[1:-1])
    I += 0.5 * dtau * (X[0] + X[-1])
    return I
        
def integration(X_tau):
    dtau = X_tau.mesh.beta / (len(X_tau.data) - 1)
    return trapez(X_tau.data[:,0,0], dtau)


def convolution(X_tau, Y_tau, n_iw, indices,  statistic="Fermion"):
    X_iw = GfImFreq(indices=indices, beta=X_tau.mesh.beta, n_points=n_iw, statistic=statistic)
    Y_iw = GfImFreq(indices=indices, beta=X_tau.mesh.beta, n_points=n_iw, statistic=statistic)
    X_iw << Fourier(X_tau if X_tau.mesh.statistic == statistic else change_statistic(X_tau))
    Y_iw << Fourier(Y_tau if Y_tau.mesh.statistic == statistic else change_statistic(Y_tau))
    Z_tau = GfImTime(indices=indices, beta=X_tau.mesh.beta, n_points=(len(X_tau.data)),
                     statistic=statistic)
    Z_tau << Fourier(X_iw * Y_iw)
    return Z_tau if X_tau.mesh.statistic == statistic else change_statistic(Z_tau)

def first_order_Sigma(G_tau, U):
    n = G_tau.data[0,0,0].real + 1
    return U * (n - 0.5)

def second_order_Sigma(G_tau, U, only_skeleton=False):
    Sigma_tau = G_tau.copy()
    G_minus_tau = reverse_tau(G_tau)
    Sigma_tau << -U**2 * G_tau * G_tau * G_minus_tau
    # non-skeleton contributions
    Hartree = 0
    if not only_skeleton:
        Hartree = U * first_order_Sigma(G_tau, U) * integration(G_minus_tau * G_tau)
    return Fourier(Sigma_tau) + Hartree

def third_order_Sigma(G_tau, U, indices, n_iw, only_skeleton=False):
    Sigma_tau = G_tau.copy()
    G_minus_tau = reverse_tau(G_tau)
    # skeleton contributions 3a and 3b
    Sigma = U**3 * G_tau * convolution(G_tau * G_minus_tau, G_tau * G_minus_tau,  n_iw, indices, "Boson")
    Sigma +=  U**3 * G_minus_tau * convolution(G_tau * G_tau, G_tau * G_tau,  n_iw, indices, "Boson")
    # non-skeleton contributions
    Hartree = 0
    if not only_skeleton:
        tadpole = first_order_Sigma(G_tau, U)
        # Diagrams 3c and 3e
        X_tau = convolution(G_tau, G_tau, n_iw, indices)
        Sigma += -tadpole * U**2 * G_tau * G_minus_tau * X_tau * 2
        # Diagram 3d
        Sigma += -tadpole * U**2 * G_tau * G_tau * reverse_tau(X_tau)
        # Hartree diagrams 3a, 3b, 3c
        Hartree += tadpole * U**2 * integration(G_minus_tau * G_tau)**2
        Hartree += tadpole**2 * U * integration(G_minus_tau * X_tau)
        X_tau = convolution(G_tau * G_tau * G_minus_tau, G_tau, n_iw, indices)
        Hartree += -U**3 * integration(G_minus_tau * X_tau)
    Sigma_tau << Sigma
    return Fourier(Sigma_tau) + Hartree
        
def generate_G_0(eps, Delta_iw):
    return inverse(iOmega_n - eps - Delta_iw)

def solve_Dyson_for_G(eps, Delta_iw, Sigma_iw):
    return inverse(iOmega_n - eps - Delta_iw - Sigma_iw)

def plot_solutions(AIM, plot_param):
    """
    For PT solver plot a sample of the results
    """
    if AIM.basis == "tau": 
        x_axis = AIM.imtime_target
    if AIM.basis == "legendre":
        x_axis = AIM.legendre_target
        
    rank = get_mpi_rank()
    if rank == plot_param["chosen_rank"]:
        fig,axes = plt.subplots(1)
        for index in range(0, plot_param["sample_max"]):
            label="Sample"+index
            axes = plot_from_csv(plot_param["file_to_read"],
                                 x_axis,
                                 index,
                                 label,
                                 axes)
        axes.legend()
        axes.set_title("PT solver on core: " + str(rank) + "")
        if AIM.basis == "tau": 
            axes.set_xlabel(r"$\tau$")
            axes.set_ylabel(r"$G(\tau)$")
        if AIM.basis == "legendre":
            axes.set_xlabel(r"l")
            axes.set_ylabel(r"$G_l$")        
        plt.show()
        
