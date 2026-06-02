import numpy as np
import scipy.constants as const
from qutip import *
from typing import Tuple, Dict


class GeSiSpinQubitSimulator:
    """Simulation framework for hole spin qubits in Ge/Si heterostructures."""
    
    def __init__(self, g_factor: float = 1.5, gamma_relax: float = 0.0):
        self.mu_B = const.physical_constants['Bohr magneton'][0]
        self.hbar = const.hbar
        self.g_factor = g_factor
        self.gamma_relax = gamma_relax  # relaxation rate
    
    def simulate_rabi(self, 
                     Omega_R: float = 2*np.pi*100e6,
                     T2_star: float = 84e-9,
                     duration: float = 400e-9,
                     n_points: int = 3000) -> Dict:
        
        H_drive = Omega_R * sigmax() / 2
        def drive_coeff(t, args):
            return np.cos(args['omega'] * t)
        
        H = [0 * sigmaz(), [H_drive, drive_coeff]]
        
        psi0 = basis(2, 0)  # Start in |↑⟩
        gamma_dephase = 1.0 / T2_star
        c_ops = [np.sqrt(gamma_dephase) * sigmaz()]
        if self.gamma_relax > 0:
            c_ops.append(np.sqrt(self.gamma_relax) * sigmam())
        
        tlist = np.linspace(0, duration, n_points)
        result = mesolve(H, psi0, tlist, c_ops,
                        e_ops=[sigmax(), sigmay(), sigmaz()],
                        args={'omega': Omega_R})
        
        p_up = (1 + result.expect[2]) / 2
        
        return {
            'time': tlist,
            'p_up': p_up,
            'sz': result.expect[2],
            'Omega_R': Omega_R,
            'T2_star': T2_star
        }

    def simulate_ramsey(self, 
                       T2_star: float = 84e-9,
                       duration: float = 250e-9,
                       n_points: int = 2000) -> Dict:
        
        tlist = np.linspace(0, duration, n_points)
        psi0 = (basis(2,0) + basis(2,1)).unit()
        
        gamma = 1.0 / T2_star
        c_ops = [np.sqrt(gamma) * sigmaz()]
        
        result = mesolve(0 * sigmaz(), psi0, tlist, c_ops, e_ops=[sigmax()])
        
        return {'time': tlist, 'contrast': result.expect[0]}
