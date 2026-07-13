"""
Schrödinger Equation Implementation

i ψ_t = -½ ∇²ψ + V(x) ψ

Complex-valued solution split into real/imaginary parts.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Callable
import math

from .base import PDE, Domain, BoundaryCondition, InitialCondition


class SchrodingerEquation(PDE):
    """
    Time-dependent Schrödinger equation.
    
    Network outputs 2 channels: [ψ_real, ψ_imag]
    """
    
    def __init__(
        self,
        domain: Domain,
        potential: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
        hbar: float = 1.0,
        mass: float = 1.0,
    ):
        parameters = {"hbar": hbar, "mass": mass}
        super().__init__(domain, boundary_conditions, initial_conditions, parameters)
        self.potential = potential or (lambda **kwargs: torch.zeros_like(list(kwargs.values())[0]))
        self.hbar = hbar
        self.mass = mass
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Use SchrodingerEquation1D or 2D")


class SchrodingerEquation1D(SchrodingerEquation):
    """
    1D Schrödinger: i ψ_t = -½ ψ_xx + V(x) ψ
    
    Real/Imaginary split:
    -ψ_imag_t = -½ ψ_real_xx + V ψ_real
    ψ_real_t = -½ ψ_imag_xx + V ψ_imag
    """
    
    def __init__(
        self,
        x_min: float = -5.0,
        x_max: float = 5.0,
        t_min: float = 0.0,
        t_max: float = 2.0,
        potential: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
        hbar: float = 1.0,
        mass: float = 1.0,
    ):
        domain = Domain(bounds={"x": (x_min, x_max), "t": (t_min, t_max)})
        
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
            ]
        
        if initial_conditions is None:
            def gaussian_packet(x):
                k0 = 5.0
                sigma = 0.5
                x0 = -1.0
                psi_real = torch.exp(-(x - x0)**2 / (2 * sigma**2)) * torch.cos(k0 * x)
                psi_imag = torch.exp(-(x - x0)**2 / (2 * sigma**2)) * torch.sin(k0 * x)
                norm = (2 * math.pi * sigma**2)**0.25
                return torch.stack([psi_real / norm, psi_imag / norm], dim=-1)
            
            initial_conditions = [
                InitialCondition({"t": t_min}, gaussian_packet)
            ]
        
        super().__init__(domain, potential, boundary_conditions, initial_conditions, hbar, mass)
        self.x_min, self.x_max = x_min, x_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute Schrödinger residual for complex wavefunction."""
        x = torch.cat([coords['x'], coords['t']], dim=-1)
        psi = model(x)  # Shape: (N, 2) - [real, imag]
        
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Time derivatives
        psi_real_t = torch.autograd.grad(
            psi_real.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        psi_imag_t = torch.autograd.grad(
            psi_imag.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Spatial second derivatives
        psi_real_x = torch.autograd.grad(
            psi_real.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        psi_real_xx = torch.autograd.grad(
            psi_real_x.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        
        psi_imag_x = torch.autograd.grad(
            psi_imag.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        psi_imag_xx = torch.autograd.grad(
            psi_imag_x.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        
        # Potential
        V = self.potential(coords['x'])
        
        # Schrödinger equation split:
        # i ψ_t = -½ ψ_xx + V ψ
        # Real: -ψ_imag_t = -½ ψ_real_xx + V ψ_real
        # Imag: ψ_real_t = -½ ψ_imag_xx + V ψ_imag
        
        R_real = -psi_imag_t + 0.5 * psi_real_xx - V * psi_real
        R_imag = psi_real_t + 0.5 * psi_imag_xx - V * psi_imag
        
        # Combined residual
        residual = R_real**2 + R_imag**2
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        """Analytical solution for free particle Gaussian wave packet."""
        x = coords['x']
        t = coords['t']
        
        k0 = 5.0
        sigma = 0.5
        x0 = -1.0
        hbar = self.hbar
        m = self.mass
        
        # Time evolution of Gaussian wave packet
        sigma_t_sq = sigma**2 + 1j * hbar * t / (2 * m * sigma**2)
        sigma_t = torch.sqrt(sigma_t_sq + 0j)
        
        x_center = x0 + hbar * k0 * t / m
        
        exponent = -(x - x_center)**2 / (4 * sigma * sigma_t) + 1j * (k0 * x - k0**2 * t / (2 * m))
        
        psi = torch.exp(exponent)
        norm = (2 * math.pi * sigma_t_sq)**0.25
        psi = psi / norm
        
        return torch.stack([psi.real, psi.imag], dim=-1)
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n = n_per_boundary
        
        t_left = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_left = torch.full_like(t_left, self.x_min)
        
        t_right = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_right = torch.full_like(t_right, self.x_max)
        
        x = torch.cat([x_left, x_right], dim=0).requires_grad_(True)
        t = torch.cat([t_left, t_right], dim=0).requires_grad_(True)
        
        return {"x": x, "t": t}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        t = torch.full_like(x, self.t_min)
        
        return {"x": x.requires_grad_(True), "t": t.requires_grad_(True)}


__all__ = [
    "SchrodingerEquation",
    "SchrodingerEquation1D",
]