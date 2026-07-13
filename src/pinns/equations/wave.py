"""
Wave Equation Implementation

u_tt = c² ∇²u

1D: u_tt = c² u_xx
2D: u_tt = c² (u_xx + u_yy)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional
import math

from .base import PDE, Domain, BoundaryCondition, InitialCondition


class WaveEquation(PDE):
    """
    General wave equation: u_tt = c² ∇²u
    """
    
    def __init__(
        self,
        domain: Domain,
        c: float = 1.0,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        parameters = {"c": c}
        super().__init__(domain, boundary_conditions, initial_conditions, parameters)
        self.c = c


class WaveEquation1D(WaveEquation):
    """
    1D Wave Equation: u_tt = c² u_xx
    
    Domain: x ∈ [0, 1], t ∈ [0, T]
    
    Analytical solution (d'Alembert):
    u(x,t) = f(x - ct) + g(x + ct)
    
    For zero Dirichlet BCs and sinusoidal IC:
    u(x,0) = sin(πx), u_t(x,0) = 0
    Solution: u(x,t) = sin(πx) cos(cπt)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        c: float = 1.0,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        domain = Domain(bounds={"x": (x_min, x_max), "t": (t_min, t_max)})
        
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
            ]
        
        if initial_conditions is None:
            initial_conditions = [
                InitialCondition({"t": t_min}, lambda x: torch.sin(math.pi * x)),
                InitialCondition(
                    {"t": t_min},
                    lambda x: torch.zeros_like(x),
                    # This represents u_t(x,0) = 0
                    # We'll need special handling for initial velocity
                ),
            ]
        
        super().__init__(domain, boundary_conditions, initial_conditions, {"c": c})
        self.c = c
        self.x_min, self.x_max = x_min, x_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute wave equation residual: u_tt - c² u_xx = 0"""
        x = torch.cat([coords['x'], coords['t']], dim=-1)
        u = model(x)
        
        # Second time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Second spatial derivative
        u_x = torch.autograd.grad(
            u.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        
        residual = u_tt - self.c**2 * u_xx
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        """Analytical solution: sin(πx) cos(cπt)"""
        x = coords['x']
        t = coords['t']
        return torch.sin(math.pi * x) * torch.cos(self.c * math.pi * t)
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on x=0 and x=1 boundaries."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n = n_per_boundary
        
        # Left boundary
        t_left = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_left = torch.full_like(t_left, self.x_min)
        
        # Right boundary
        t_right = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_right = torch.full_like(t_right, self.x_max)
        
        x = torch.cat([x_left, x_right], dim=0).requires_grad_(True)
        t = torch.cat([t_left, t_right], dim=0).requires_grad_(True)
        
        return {"x": x, "t": t}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Sample points at t=0 for both u and u_t."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        t = torch.full_like(x, self.t_min)
        
        return {"x": x.requires_grad_(True), "t": t.requires_grad_(True)}


class WaveEquation2D(WaveEquation):
    """
    2D Wave Equation: u_tt = c² (u_xx + u_yy)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        c: float = 1.0,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        domain = Domain(bounds={
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "t": (t_min, t_max),
        })
        
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
                BoundaryCondition("dirichlet", {"y": y_min}, 0.0),
                BoundaryCondition("dirichlet", {"y": y_max}, 0.0),
            ]
        
        if initial_conditions is None:
            initial_conditions = [
                InitialCondition(
                    {"t": t_min},
                    lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y)
                ),
            ]
        
        super().__init__(domain, boundary_conditions, initial_conditions, {"c": c})
        self.c = c
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute 2D wave equation residual: u_tt - c²(u_xx + u_yy) = 0"""
        x = torch.cat([coords['x'], coords['y'], coords['t']], dim=-1)
        u = model(x)
        
        # Second time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Spatial Laplacian
        laplacian = 0
        for dim in ['x', 'y']:
            u_d = torch.autograd.grad(
                u.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            u_dd = torch.autograd.grad(
                u_d.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            laplacian = laplacian + u_dd
        
        residual = u_tt - self.c**2 * laplacian
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        """Analytical solution for sin(πx)sin(πy) IC."""
        x = coords['x']
        y = coords['y']
        t = coords['t']
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(self.c * math.pi * math.sqrt(2) * t)
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on all four boundaries."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n = n_per_boundary
        coords_list = []
        
        # x = x_min
        x = torch.full((n, 1), self.x_min, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # x = x_max
        x = torch.full((n, 1), self.x_max, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # y = y_min
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_min, device=device)
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # y = y_max
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_max, device=device)
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        x_all = torch.cat([c[0] for c in coords_list], dim=0).requires_grad_(True)
        y_all = torch.cat([c[1] for c in coords_list], dim=0).requires_grad_(True)
        t_all = torch.cat([c[2] for c in coords_list], dim=0).requires_grad_(True)
        
        return {"x": x_all, "y": y_all, "t": t_all}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Sample points at t=0."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.full_like(x, self.t_min)
        
        return {
            "x": x.requires_grad_(True),
            "y": y.requires_grad_(True),
            "t": t.requires_grad_(True),
        }


__all__ = [
    "WaveEquation",
    "WaveEquation1D",
    "WaveEquation2D",
]