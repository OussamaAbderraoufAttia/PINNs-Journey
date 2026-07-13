"""
Equations Package

PDE and ODE definitions with analytical solutions, boundary conditions,
and residual computations.
"""

from .base import Equation, ODE, PDE
from .registry import EQUATION_REGISTRY, get_equation, register_equation
from .heat import HeatEquation, HeatEquation1D, HeatEquation2D
from .burgers import BurgersEquation, BurgersEquation1D
from .wave import WaveEquation, WaveEquation1D, WaveEquation2D
from .poisson import PoissonEquation, PoissonEquation1D, PoissonEquation2D
from .reaction_diffusion import ReactionDiffusionEquation
from .schrodinger import SchrodingerEquation
from .ode import ExponentialDecayODE, LogisticODE, HarmonicOscillatorODE

__all__ = [
    # Base classes
    "Equation",
    "ODE",
    "PDE",
    # Registry
    "EQUATION_REGISTRY",
    "get_equation",
    "register_equation",
    # Equations
    "HeatEquation",
    "HeatEquation1D",
    "HeatEquation2D",
    "BurgersEquation",
    "BurgersEquation1D",
    "WaveEquation",
    "WaveEquation1D",
    "WaveEquation2D",
    "PoissonEquation",
    "PoissonEquation1D",
    "PoissonEquation2D",
    "ReactionDiffusionEquation",
    "SchrodingerEquation",
    "ExponentialDecayODE",
    "LogisticODE",
    "HarmonicOscillatorODE",
]