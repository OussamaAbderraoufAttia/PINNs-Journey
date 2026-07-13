"""
ODE Examples for PINNs

Simple ODEs to demonstrate PINN methodology.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Callable
import math

from .base import ODE, Domain, InitialCondition


class ExponentialDecayODE(ODE):
    """
    Exponential Decay: dy/dt = -λ y
    
    Analytical solution: y(t) = y₀ e^(-λt)
    
    This is the simplest PINN example - 1D input, 1D output.
    """
    
    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 2.0,
        decay_rate: float = 1.0,
        initial_condition: Optional[Callable] = None,
        y0: float = 1.0,
    ):
        domain = Domain(bounds={"t": (t_min, t_max)})
        
        if initial_condition is None:
            initial_condition = InitialCondition(
                {"t": t_min},
                y0
            )
        
        super().__init__(domain, [initial_condition] if initial_condition else [], {"decay_rate": decay_rate})
        self.decay_rate = decay_rate
        self.t_min, self.t_max = t_min, t_max
        self.y0 = y0
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute ODE residual: dy/dt + λ y = 0"""
        t = coords['t']
        y = model(t)
        
        dy_dt = torch.autograd.grad(
            y.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        residual = dy_dt + self.decay_rate * y
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        t = coords['t']
        return self.y0 * torch.exp(-self.decay_rate * t)
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        t = torch.full((n, 1), self.t_min, device=device)
        return {"t": t.requires_grad_(True)}


class LogisticODE(ODE):
    """
    Logistic Equation: dy/dt = r y (1 - y/K)
    
    Analytical solution: y(t) = K / (1 + (K/y₀ - 1) e^(-rt))
    """
    
    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 5.0,
        r: float = 1.0,
        K: float = 1.0,
        y0: float = 0.1,
    ):
        domain = Domain(bounds={"t": (t_min, t_max)})
        
        initial_condition = InitialCondition(
            {"t": t_min},
            y0
        )
        
        super().__init__(domain, [initial_condition], {"r": r, "K": K})
        self.r = r
        self.K = K
        self.y0 = y0
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute logistic ODE residual: dy/dt - r y (1 - y/K) = 0"""
        t = coords['t']
        y = model(t)
        
        dy_dt = torch.autograd.grad(
            y.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        residual = dy_dt - self.r * y * (1 - y / self.K)
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        t = coords['t']
        return self.K / (1 + (self.K / self.y0 - 1) * torch.exp(-self.r * t))
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.full((n, 1), self.t_min, device=device)
        return {"t": t.requires_grad_(True)}


class HarmonicOscillatorODE(ODE):
    """
    Harmonic Oscillator: d²y/dt² + ω² y = 0
    
    Converted to first-order system:
    dy/dt = v
    dv/dt = -ω² y
    
    Network outputs [y, v]
    Analytical: y(t) = A cos(ωt) + B sin(ωt)
    """
    
    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 10.0,
        omega: float = 2.0,
        y0: float = 1.0,
        v0: float = 0.0,
    ):
        domain = Domain(bounds={"t": (t_min, t_max)})
        
        # Initial conditions for both y and v
        initial_conditions = [
            InitialCondition({"t": t_min}, lambda t: torch.tensor([[y0, v0]]).to(t.device)),
        ]
        
        super().__init__(domain, initial_conditions, {"omega": omega})
        self.omega = omega
        self.y0 = y0
        self.v0 = v0
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute harmonic oscillator residual as first-order system"""
        t = coords['t']
        yv = model(t)  # (N, 2) - [y, v]
        
        y = yv[:, 0:1]
        v = yv[:, 1:2]
        
        dy_dt = torch.autograd.grad(
            y.sum(), t, create_graph=True, retain_graph=True
        )[0]
        dv_dt = torch.autograd.grad(
            v.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        # System: dy/dt = v, dv/dt = -ω²y
        R1 = dy_dt - v
        R2 = dv_dt + self.omega**2 * y
        
        residual = R1**2 + R2**2
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        t = coords['t']
        y = self.y0 * torch.cos(self.omega * t) + self.v0 / self.omega * torch.sin(self.omega * t)
        v = -self.y0 * self.omega * torch.sin(self.omega * t) + self.v0 * torch.cos(self.omega * t)
        return torch.cat([y, v], dim=-1)
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.full((n, 1), self.t_min, device=device)
        return {"t": t.requires_grad_(True)}


class PendulumODE(ODE):
    """
    Nonlinear Pendulum: d²θ/dt² + (g/L) sin(θ) = 0
    
    First-order system:
    dθ/dt = ω
    dω/dt = -(g/L) sin(θ)
    
    Network outputs [θ, ω]
    """
    
    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 10.0,
        g_over_L: float = 9.81,
        theta0: float = 0.5,  # radians
        omega0: float = 0.0,
    ):
        domain = Domain(bounds={"t": (t_min, t_max)})
        
        initial_conditions = [
            InitialCondition({"t": t_min}, lambda t: torch.tensor([[theta0, omega0]]).to(t.device)),
        ]
        
        super().__init__(domain, initial_conditions, {"g_over_L": g_over_L})
        self.g_over_L = g_over_L
        self.theta0 = theta0
        self.omega0 = omega0
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        t = coords['t']
        theta_omega = model(t)
        
        theta = theta_omega[:, 0:1]
        omega = theta_omega[:, 1:2]
        
        dtheta_dt = torch.autograd.grad(
            theta.sum(), t, create_graph=True, retain_graph=True
        )[0]
        domega_dt = torch.autograd.grad(
            omega.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        R1 = dtheta_dt - omega
        R2 = domega_dt + self.g_over_L * torch.sin(theta)
        
        return R1**2 + R2**2
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        # No simple analytical solution for nonlinear pendulum
        return None
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.full((n, 1), self.t_min, device=device)
        return {"t": t.requires_grad_(True)}


from ..registry import register_equation
register_equation("exponential_decay", ExponentialDecayODE)
register_equation("logistic", LogisticODE)
register_equation("harmonic_oscillator", HarmonicOscillatorODE)
register_equation("pendulum", PendulumODE)


__all__ = [
    "ExponentialDecayODE",
    "LogisticODE",
    "HarmonicOscillatorODE",
    "PendulumODE",
]