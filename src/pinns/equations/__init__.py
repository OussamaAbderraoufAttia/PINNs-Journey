"""
Equations Package

PDE and ODE definitions with analytical solutions, boundary conditions,
and residual computations.
"""

from .base import Equation, ODE, PDE, Domain, BoundaryCondition, InitialCondition
from .registry import EQUATION_REGISTRY, get_equation, register_equation, list_equations

__all__ = [
    "Equation",
    "ODE", 
    "PDE",
    "Domain",
    "BoundaryCondition",
    "InitialCondition",
    "EQUATION_REGISTRY",
    "get_equation",
    "register_equation",
    "list_equations",
]