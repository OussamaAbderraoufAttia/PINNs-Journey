"""
Equation Registry

Central registry for all available equations with easy lookup.
"""

from typing import Dict, Type, Any, Optional
from .base import Equation


EQUATION_REGISTRY: Dict[str, Type[Equation]] = {}


def register_equation(name: str, equation_class: Type[Equation] = None):
    """
    Register an equation class.
    
    Can be used as decorator:
        @register_equation("my_equation")
        class MyEquation(Equation): ...
    
    Or as function:
        register_equation("my_equation", MyEquation)
    """
    def decorator(cls: Type[Equation]) -> Type[Equation]:
        EQUATION_REGISTRY[name.lower()] = cls
        return cls
    
    if equation_class is not None:
        EQUATION_REGISTRY[name.lower()] = equation_class
        return equation_class
    
    return decorator


def get_equation(name: str, **kwargs) -> Equation:
    """
    Get equation instance by name.
    
    Args:
        name: Equation name (case-insensitive)
        **kwargs: Arguments passed to equation constructor
    
    Returns:
        Equation instance
    """
    name = name.lower()
    if name not in EQUATION_REGISTRY:
        available = ", ".join(sorted(EQUATION_REGISTRY.keys()))
        raise ValueError(f"Unknown equation: {name}. Available: {available}")
    
    return EQUATION_REGISTRY[name](**kwargs)


def list_equations() -> list:
    """List all registered equation names."""
    return sorted(EQUATION_REGISTRY.keys())


# Import all equations to register them
from . import (
    heat,
    burgers,
    wave,
    poisson,
    reaction_diffusion,
    schrodinger,
    ode,
)

# Explicitly register base equations
from .heat import HeatEquation, HeatEquation1D, HeatEquation2D
from .burgers import BurgersEquation, BurgersEquation1D
from .wave import WaveEquation, WaveEquation1D, WaveEquation2D
from .poisson import PoissonEquation, PoissonEquation1D, PoissonEquation2D
from .reaction_diffusion import ReactionDiffusionEquation
from .schrodinger import SchrodingerEquation
from .ode import ExponentialDecayODE, LogisticODE, HarmonicOscillatorODE

register_equation("heat", HeatEquation)
register_equation("heat_1d", HeatEquation1D)
register_equation("heat_2d", HeatEquation2D)
register_equation("burgers", BurgersEquation)
register_equation("burgers_1d", BurgersEquation1D)
register_equation("wave", WaveEquation)
register_equation("wave_1d", WaveEquation1D)
register_equation("wave_2d", WaveEquation2D)
register_equation("poisson", PoissonEquation)
register_equation("poisson_1d", PoissonEquation1D)
register_equation("poisson_2d", PoissonEquation2D)
register_equation("reaction_diffusion", ReactionDiffusionEquation)
register_equation("schrodinger", SchrodingerEquation)
register_equation("exponential_decay", ExponentialDecayODE)
register_equation("logistic", LogisticODE)
register_equation("harmonic_oscillator", HarmonicOscillatorODE)

__all__ = [
    "EQUATION_REGISTRY",
    "register_equation",
    "get_equation",
    "list_equations",
]