import numpy as np
import time
from triqs.gf import *
from triqs.dos import *
from triqs.plot.mpl_interface import oplot, plt
from laim.utility.db_gen import Delta_from_lists, save_gf, get_data
from laim.utility.mpi_tools import * 
from laim.utility.tools import plot_from_csv, name, del_file
from scipy import interpolate

class SC_solver():
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
        self.target_n_tau = solver_params["target_n_tau"]
        self.tau_arr = solver_params["tau_arr"]
        self.writing = solver_params["writing"]
        
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
        self.legendre_target =  np.linspace(1, self.n_l, self.n_l, endpoint = True)


    def solve(self, param, delta_inp):
        """
        Solve the AIM using PT
        """
        self.eps = param["eps"]
        self.U = param["U"]
        self.delta = delta_inp        
        
        P_0 = generate_P_0(self.eps, self.U, self.tau_arr)
        G_0 = get_gf(P_0)
        P = get_propagator(self.eps, self.U, self.delta, self.tau_arr, self_consistent=True, order=4)
        G = get_gf(P)
        tau_g = np.linspace(0, self.beta, self.target_n_tau+1, endpoint=True)
        tau_arr_new, gf_new = interpolate_NCA(tau_g, G[0], self.n_tau)
        self.G_tr = gf_to_triqs(gf_new,self.beta,tau_arr_new,"Gtau")        
        self.G_iw << Fourier(self.G_tr)
        self.G_l << MatsubaraToLegendre(self.G_iw)
        
    def write_data(self):
        only_gf=True
        tau_filename = name("G_SC", self.beta, "tau")
        leg_filename = name("G_SC", self.beta, "legendre")

        save_gf(leg_filename, self.G_l, "legendre",
                self.target_n_tau, only_gf=only_gf)
        save_gf(tau_filename, self.G_tr, "tau",
                self.target_n_tau, only_gf=only_gf)                
            
# Generating P_0
def generate_P_0(eps, U, tau):
    P_0 = np.zeros((4, len(tau)))
    energies = [0, eps - U / 2, eps - U / 2, 2 * eps]
    for p, e in zip(P_0, energies):
        p[:] = np.exp(-tau * e)
    return P_0

# Propagator self energy

def generate_self_energy(P, Delta):
    n_tau = len(Delta)
    Q = np.zeros((4, n_tau))
    Q[0] = -(P[1] + P[2]) * np.flipud(Delta)
    Q[1] = -P[0] * Delta - P[3] * np.flipud(Delta)
    Q[2] = -P[0] * Delta - P[3] * np.flipud(Delta)
    Q[3] = -(P[1] + P[2]) * Delta
    return Q

# Solving Volterra equation

def second_order_w(n):
    w = np.zeros((n, n))
    # trapezoid rule
    for i in range(1, n):
        w[i, 0] = 0.5
        w[i, 1:i] = 1
        w[i, i] = 0.5
    return w

def fourth_order_w(n):
    w = np.zeros((n, n))
    # i = 1 (trapezoid rule)
    w[1, 0:2] = 1 / 2
    # i = 2 (Simpson's rule)
    w[2, 0] = 1 / 3
    w[2, 1] = 4 / 3
    w[2, 2] = 1 / 3
    # i = 3 (Simpson's 3/8 rule)
    w[3, 0] = 3 / 8
    w[3, 1:3] = 9 / 8
    w[3, 3] = 3 / 8
    # i = 4 (composite Simpson's rule)
    w[4, 0] = 1 / 3
    w[4, 1] = 4 / 3
    w[4, 2] = 2 / 3
    w[4, 3] = 4 / 3
    w[4, 4] = 1 / 3
    # i >= 5 (fourth-order Gregory's rule)
    for i in range(5, n):
        w[i, 0] = 3 / 8
        w[i, 1] = 7 / 6
        w[i, 2] = 23 / 24
        w[i, 3:i-2] = 1
        w[i, i - 2] = 23 / 24
        w[i, i - 1] = 7 / 6
        w[i, i] = 3 / 8
    return w

def solve_Volterra(y_0, f, k, dx, scheme_order=4):
    """ Solve Volterra Integral-Differential Equation:
    y'(x) = f(x) y(x) + int_0_x dx' k(x - x') y(x')
    """
    n_x = len(f)
    if(scheme_order == 2):
        dw = dx * second_order_w(n_x)
    if(scheme_order == 4):
        dw = dx * fourth_order_w(n_x)
    y = np.zeros(n_x)
    yx = np.zeros(n_x)
    # i = 0 (initial condition)
    y[0] = y_0
    yx[0] = f[0] * y_0
    if(scheme_order == 4):
        # i = 1, 2 (Simpson's rule with middle point + quadratic interpolation)
        k12 = 3 / 8 * k[0] + 3 / 4 * k[1] - 1 / 8 * k[2]
        A = np.array([[1, -2 / 3 * dx, 0, dx / 12],
                      [-f[1] - dx * k12 / 2 - dx * k[0] / 6, 1, dx * k12 / 12, 0],
                      [0, -4 / 3 * dx, 1, -dx / 3],
                      [-4 / 3 * dx * k[1], 0, -f[2] - dx * k[0] / 3, 1]])
        B = np.array([y[0] + 5 / 12 * dx * yx[0],
                      (dx * k[1] / 6 + dx * k12 / 4) * y[0],
                      y[0] + dx * yx[0] / 3,
                      dx * k[2] * y[0] / 3])
        y[1], yx[1], y[2], yx[2] = np.linalg.solve(A, B)
    for i in range(3 if scheme_order == 4 else 1, n_x):
        j_start = max(i - (5 if scheme_order == 4 else 1), 0)
        # int_1 = sum((dw[i, j] - dw[i-1, j]) * yx[j] for j in range(j_start, i))
        # int_2 = sum(dw[i, j] * k[i-j] * y[j] for j in range(i))
        int_1 = (dw[i, j_start:i] - dw[i-1, j_start:i]) @ yx[j_start:i]
        int_2 = (dw[i, :i] * np.flipud(k[1:i+1])) @ y[:i]
        y[i] = y[i-1] + int_1 + dw[i, i] * int_2
        y[i] /= 1 - dw[i, i] * f[i] - (dw[i, i])**2 * k[0]
        yx[i] = f[i] * y[i] + int_2 + dw[i, i] * k[0] * y[i]
    return y, yx

# Dyson equation for the NCA propagator

def solve_Dyson_for_P(eps, U, Q, tau, order=4):
    dtau = tau[-1] / (len(tau) - 1)
    P = np.zeros((4, len(tau)))
    dP = np.zeros((4, len(tau)))
    energies = [0, eps - U / 2, eps - U / 2, 2 * eps]
    for p, dp, q, e in zip(P, dP, Q, energies):
        # y(x) = P(tau), y(0) = 1, f(x) = -energy, k(x) = Q(tau)
        f = np.full((len(tau),), -e)
        p[:], dp[:] = solve_Volterra(1, f, q, dtau, scheme_order=order)
    return P, dP


def get_propagator(eps, U, Delta, tau, self_consistent=True, order=4):
    tol, n_iter_max = 1e-5, 40
    #tol, n_iter_max = 1e-5, 1
    P_prev = generate_P_0(eps, U, tau)
    for i in range(n_iter_max):
        Q = generate_self_energy(P_prev, Delta)
        P = solve_Dyson_for_P(eps, U, Q, tau, order=order)[0]    
        if np.allclose(P_prev, P, atol=tol) or not self_consistent:
            #print("Converged in iteration {}".format(i))
            return P
        else:
            P_prev[:] = 0.8 * P + 0.2 * P_prev
    #print("Solution not converged!")
    return P
# Green functions and static observables

def get_gf(P):
    G = np.zeros((2, P.shape[1]))
    Z = np.sum(P[:, -1])
    G[0, :] = -(np.flipud(P[0, :]) * P[1, :] + np.flipud(P[1, :]) * P[3, :]) / Z
    G[1, :] = -(np.flipud(P[0, :]) * P[2, :] + np.flipud(P[2, :]) * P[3, :]) / Z
    return G

def expectation_value(A, P):
    Z = np.sum(P[:, -1])
    return np.sum(P[:, -1] * A[:]) / Z

def gf_to_triqs(g,b_v,tau,label):
    #mesh_iw = MeshImFreq(beta, 'Fermion', n_max=1000)
    gf=GfImTime(indices=[0], beta=b_v, n_points = len(tau), name=label)
    gf.data[:,0,0] = g
    #get_obj_prop(gf)
    return gf

def interpolate_NCA(tau, gf, new_n_tau):
    tck = interpolate.splrep(tau, gf, s=0.000001)
    new_tau = np.linspace(0, tau[-1], num=new_n_tau, endpoint=True)
    new_gf = interpolate.splev(new_tau, tck, der=0)
    return new_tau, new_gf
