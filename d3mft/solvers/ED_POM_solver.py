# LAIM -- Learning the Anderson Impurity Model
# Copyright (C) 2021 King's College London 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import time
from triqs.gf import *
from triqs.dos import *
from triqs.operators import Operator, c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from pomerol2triqs import PomerolED
from triqs.plot.mpl_interface import oplot, plt
from utility.db_gen import Delta_from_lists, save_gf
from utility.mpi_tools import * 
from utility.tools import plot_from_csv, get_obj_prop
from itertools import product

class ED_POM_solver():
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
        self.target_n_tau = solver_params["target_n_tau"]
        self.energy_window = solver_params["energy_window"]
        self.n_w = solver_params["n_w"]

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
        self.imtime_target = np.linspace(0, self.beta, 
                                         self.target_n_tau+1, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)        
        self.legendre_target = np.linspace(1, self.n_l, self.n_l, endpoint = True)




    def solve_dirty(self, param):
        """
         Solve the AIM using ED pomerol
        """
        e_list = param["e_list"]
        V_list = param["V_list"]
        self.eps = param["eps"]
        self.U = param["U"]
        mu = self.eps
        U = self.U
        beta = self.beta
        n_iw = self.n_iw
        n_tau = self.n_tau
        n_w = self.n_w
        energy_window = self.energy_window
        spin_names = ("up", "dn")
        # GF structure
        gf_struct = [['up', [0]], ['dn', [0]]]
        # Conversion from TRIQS to Pomerol notation for operator indices
        index_converter = {}
        index_converter.update({(sn, 0) : ("loc", 0, "down" if sn == "dn" else "up") for sn in spin_names})
        index_converter.update({("B%i_%s" % (k, sn), 0) : ("bath" + str(k), 0, "down" if sn == "dn" else "up")
                                for k, sn in product(list(range(len(e_list))), spin_names)})

        # Make PomerolED solver object
        ed = PomerolED(index_converter, verbose = True)

        # Number of particles on the impurity
        H_loc = mu*(n('up', 0) + n('dn', 0)) + U * (n('up', 0) -0.5) * (n('dn', 0) -0.5) 

        # Bath Hamiltonian
        H_bath = sum(eps*n("B%i_%s" % (k, sn), 0)
                     for sn, (k, eps) in product(spin_names, enumerate(e_list)))

        # Hybridization Hamiltonian
        H_hyb = Operator()
        for k, v in enumerate(V_list):
            H_hyb += sum(        v   * c_dag("B%i_%s" % (k, sn), 0) * c(sn, 0) +
                                 np.conj(v)  * c_dag(sn, 0) * c("B%i_%s" % (k, sn), 0)
                                 for sn in spin_names)
        
        # Complete Hamiltonian
        H = H_loc + H_hyb + H_bath

        # Diagonalize H
        ed.diagonalize(H)

        # Compute G(i\omega)
        G_iw = ed.G_iw(gf_struct, beta, n_iw)

        # Compute G(\tau)
        G_tau = ed.G_tau(gf_struct, beta, n_tau)
         
        # Writing results to file        
        only_gf = True
        if self.basis == "legendre": 
            self.G_l << MatsubaraToLegendre(G_iw['up'])
            save_gf(self.file_to_write, self.G_l, self.basis,
                    self.target_n_tau, only_gf=only_gf)
    
        if self.basis == "tau":
            save_gf(self.file_to_write, G_tau['up'], self.basis,
                    self.target_n_tau, only_gf=only_gf)
         
        # Compute G(\omega)        
        self.G_w = ed.G_w(gf_struct, beta, energy_window, n_w, 0.01)        
        
    def solve(self, param):
        """
        Solve the AIM using ED pomerol: ED BUG - not sure what is wrong!
        """
        self.e_list = param["e_list"]
        self.V_list = param["V_list"]
        self.eps = param["eps"]
        self.U = param["U"]
        spin_names = ("up", "dn")

        # Solving the AIM 
        self.gf_struct, self.index_converter = pom_to_triqs(self.e_list, spin_names)        
        ed = PomerolED(self.index_converter, verbose = True)
        
        # Number of particles on the impurity
        H_loc = self.eps*(n('up', 0) + n('dn', 0)) + self.U * (n('up', 0) -0.5) * (n('dn', 0) -0.5) 

        # Bath Hamiltonian
        H_bath = sum(self.eps*n("B%i_%s" % (k, sn), 0)
                     for sn, (k, eps) in product(spin_names, enumerate(self.e_list)))

        # Hybridization Hamiltonian
        H_hyb = Operator()
        for k, v in enumerate(self.V_list):
            H_hyb += sum(        v   * c_dag("B%i_%s" % (k, sn), 0) * c(sn, 0) +
                                 np.conj(v)  * c_dag(sn, 0) * c("B%i_%s" % (k, sn), 0)
                                 for sn in spin_names)
        
        H = H_loc + H_hyb + H_bath        
        ed.diagonalize(H)
        
        self.G_iw = ed.G_iw(self.gf_struct, self.beta, self.n_iw)
        self.G_tau = ed.G_tau(self.gf_struct, self.beta, self.n_tau)

        # Writing results to file        
        only_gf = True
        if self.basis == "legendre": 
            self.G_l << MatsubaraToLegendre(self.G_iw['up'])
            save_gf(self.file_to_write, self.G_l, self.basis,
                    self.target_n_tau, only_gf=only_gf)
    
        if self.basis == "tau":
            save_gf(self.file_to_write, self.G_tau['up'], self.basis,
                    self.target_n_tau, only_gf=only_gf)

        # Compute G(\omega)
        # self.G_w = self.ed.G_w(self.gf_struct, self.beta, self.energy_window, self.n_w, 0.01)        

    def construct_hamiltonian(self, spin_names):
        # Number of particles on the impurity
        H_loc = self.eps*(n('up', 0) + n('dn', 0)) + self.U * (n('up', 0) -0.5) * (n('dn', 0) -0.5) 
            
        # Bath Hamiltonian
        H_bath = sum(self.eps*n("B%i_%s" % (k, sn), 0)
                              for sn, (k, eps) in product(spin_names, enumerate(self.e_list)))
            
        # Hybridization Hamiltonian
        H_hyb = Operator()
        for k, v in enumerate(self.V_list):
            H_hyb += sum(v*c_dag("B%i_%s" % (k, sn), 0) * c(sn, 0) +
                         np.conj(v)  * c_dag(sn, 0) * c("B%i_%s" % (k, sn), 0)
                         for sn in spin_names)

        self.H = H_loc + H_hyb + H_bath        
        return self.H

            
def pom_to_triqs(e_list, spin_names):
    # GF structure
    gf_struct = [['up', [0]], ['dn', [0]]]
    # Conversion from TRIQS to Pomerol notation for operator indices
    index_converter = {}
    index_converter.update({(sn, 0) : ("loc", 0, "down" if sn == "dn" else "up") for sn in spin_names})
    index_converter.update({("B%i_%s" % (k, sn), 0) : ("bath" + str(k), 0, "down" if sn == "dn" else "up")
                            for k, sn in product(list(range(len(e_list))), spin_names)})
    return gf_struct, index_converter
