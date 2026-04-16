"""
amerta: 1D Saint-Venant (shallow water) equations solver.

Numerical method: MUSCL (minmod) reconstruction + HLLC Riemann solver
                  + SSP-RK2 (Heun) time integration.
"""
__version__ = "0.0.2"
__author__ = "Dasapta E. Irawan, Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Astyka Pamumpuni, Rendy D. Kartiko, Edi Riawan, Rusmawan Suwarman, Deny J. Puradimaja"
__license__ = "MIT"

from .core.solver import SaintVenantSolver
from .core.cases import CASES, get_case
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = ["SaintVenantSolver", "CASES", "get_case", "ConfigManager", "DataHandler"]
