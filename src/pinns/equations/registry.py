"""
Equation Registry

Central registry for all available equations with easy lookup.
Uses lazy imports to avoid circular dependencies.
"""

from typing import Dict, Type
from .base import Equation


EQUATION_REGISTRY: Dict[str, Type[Equation]] = {}
_INITIALIZED = False


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


def _ensure_registered():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    from . import heat, burgers, wave, poisson, reaction_diffusion, schrodinger, ode

    register_equation("heat", heat.HeatEquation)
    register_equation("heat_1d", heat.HeatEquation1D)
    register_equation("heat_2d", heat.HeatEquation2D)
    register_equation("burgers", burgers.BurgersEquation)
    register_equation("burgers_1d", burgers.BurgersEquation1D)
    register_equation("wave", wave.WaveEquation)
    register_equation("wave_1d", wave.WaveEquation1D)
    register_equation("wave_2d", wave.WaveEquation2D)
    register_equation("poisson", poisson.PoissonEquation)
    register_equation("poisson_1d", poisson.PoissonEquation1D)
    register_equation("poisson_2d", poisson.PoissonEquation2D)
    register_equation("reaction_diffusion", reaction_diffusion.ReactionDiffusionEquation)
    register_equation("reaction_diffusion_1d", reaction_diffusion.ReactionDiffusion1D)
    register_equation("reaction_diffusion_2d", reaction_diffusion.ReactionDiffusion2D)
    register_equation("schrodinger", schrodinger.SchrodingerEquation)
    register_equation("schrodinger_1d", schrodinger.SchrodingerEquation1D)
    register_equation("exponential_decay", ode.ExponentialDecayODE)
    register_equation("logistic", ode.LogisticODE)
    register_equation("harmonic_oscillator", ode.HarmonicOscillatorODE)
    register_equation("pendulum", ode.PendulumODE)


def get_equation(name: str, **kwargs) -> Equation:
    """
    Get equation instance by name.

    Args:
        name: Equation name (case-insensitive)
        **kwargs: Arguments passed to equation constructor

    Returns:
        Equation instance
    """
    _ensure_registered()
    name = name.lower()
    if name not in EQUATION_REGISTRY:
        available = ", ".join(sorted(EQUATION_REGISTRY.keys()))
        raise ValueError(f"Unknown equation: {name}. Available: {available}")
    return EQUATION_REGISTRY[name](**kwargs)


def list_equations() -> list:
    """List all registered equation names."""
    _ensure_registered()
    return sorted(EQUATION_REGISTRY.keys())


__all__ = [
    "EQUATION_REGISTRY",
    "register_equation",
    "get_equation",
    "list_equations",
]
