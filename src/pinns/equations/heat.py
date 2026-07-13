"""
Heat Equation Implementation

u_t = α ∇²u

1D: u_t = α u_xx
2D: u_t = α (u_xx + u_yy)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Any
import math

from .base import PDE, Domain, BoundaryCondition, InitialCondition


class HeatEquation(PDE):
    """
    General heat equation: u_t = α ∇²u
    
    Supports 1D, 2D, 3D through domain specification.
    """
    
    def __init__(
        self,
        domain: Domain,
        alpha: float = 0.01,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        parameters = {"alpha": alpha}
        super().__init__(domain, boundary_conditions, initial_conditions, parameters)
        self.alpha = alpha
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """
        Compute heat equation residual: u_t - α∇²u = 0
        
        Args:
            model: Neural network
            coords: Dictionary with coordinate tensors (requires_grad=True)
        
        Returns:
            Residual tensor
        """
        # Concatenate coordinates for model input
        # Assume spatial dims first, then time
        spatial_dims = [k for k in coords.keys() if k != 't']
        coord_list = [coords[d] for d in spatial_dims]
        if 't' in coords:
            coord_list.append(coords['t'])
        x = torch.cat(coord_list, dim=-1)
        
        u = model(x)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Spatial Laplacian
        laplacian = torch.zeros_like(u)
        for dim in spatial_dims:
            u_d = torch.autograd.grad(
                u.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            u_dd = torch.autograd.grad(
                u_d.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            laplacian = laplacian + u_dd
        
        # Residual: u_t - α∇²u
        residual = u_t - self.alpha * laplacian
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        """General analytical solution not available without specific IC/BC."""
        return None
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on spatial boundaries."""
        # Implementation depends on specific dimension
        raise NotImplementedError("Use HeatEquation1D or HeatEquation2D")
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Sample points at initial time."""
        # Implementation depends on specific dimension
        raise NotImplementedError("Use HeatEquation1D or HeatEquation2D")


class HeatEquation1D(HeatEquation):
    """
    1D Heat Equation: u_t = α u_xx
    
    Domain: x ∈ [x_min, x_max], t ∈ [t_min, t_max]
    
    Analytical solution for:
    - IC: u(x, 0) = sin(πx)
    - BC: u(0, t) = u(1, t) = 0
    - Solution: u(x, t) = e^(-απ²t) sin(πx)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        alpha: float = 0.01,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        domain = Domain(bounds={"x": (x_min, x_max), "t": (t_min, t_max)})
        
        # Default BCs: Dirichlet zero at boundaries
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
            ]
        
        # Default IC: sin(πx)
        if initial_conditions is None:
            initial_conditions = [
                InitialCondition({"t": t_min}, lambda x: torch.sin(math.pi * x))
            ]
        
        super().__init__(domain, boundary_conditions, initial_conditions, {"alpha": alpha})
        self.alpha = alpha
        self.x_min, self.x_max = x_min, x_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute 1D heat equation residual: u_t - α u_xx = 0"""
        x = torch.cat([coords['x'], coords['t']], dim=-1)
        u = model(x)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Second spatial derivative
        u_x = torch.autograd.grad(
            u.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        
        residual = u_t - self.alpha * u_xx
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        """Analytical solution for sin(πx) IC with zero Dirichlet BCs."""
        x = coords['x']
        t = coords['t']
        return torch.exp(-self.alpha * math.pi**2 * t) * torch.sin(math.pi * x)
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on x=0 and x=1 boundaries."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Left boundary (x=0)
        t_left = torch.rand(n_per_boundary, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_left = torch.full_like(t_left, self.x_min)
        
        # Right boundary (x=1)
        t_right = torch.rand(n_per_boundary, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x_right = torch.full_like(t_right, self.x_max)
        
        # Combine
        x = torch.cat([x_left, x_right], dim=0).requires_grad_(True)
        t = torch.cat([t_left, t_right], dim=0).requires_grad_(True)
        
        return {"x": x, "t": t}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Sample points at t=0."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        t = torch.full_like(x, self.t_min)
        
        return {"x": x.requires_grad_(True), "t": t.requires_grad_(True)}


class HeatEquation2D(HeatEquation):
    """
    2D Heat Equation: u_t = α (u_xx + u_yy)
    
    Domain: x ∈ [x_min, x_max], y ∈ [y_min, y_max], t ∈ [t_min, t_max]
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        alpha: float = 0.01,
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
                )
            ]
        
        super().__init__(domain, boundary_conditions, initial_conditions, {"alpha": alpha})
        self.alpha = alpha
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute 2D heat equation residual: u_t - α(u_xx + u_yy) = 0"""
        x = torch.cat([coords['x'], coords['y'], coords['t']], dim=-1)
        u = model(x)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Spatial derivatives
        laplacian = 0
        for dim in ['x', 'y']:
            u_d = torch.autograd.grad(
                u.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            u_dd = torch.autograd.grad(
                u_d.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            laplacian = laplacian + u_dd
        
        residual = u_t - self.alpha * laplacian
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        """Analytical solution for sin(πx)sin(πy) IC with zero Dirichlet BCs."""
        x = coords['x']
        y = coords['y']
        t = coords['t']
        return torch.exp(-2 * self.alpha * math.pi**2 * t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on all four boundaries."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n = n_per_boundary
        coords_list = []
        
        # x = x_min boundary
        x = torch.full((n, 1), self.x_min, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # x = x_max boundary
        x = torch.full((n, 1), self.x_max, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # y = y_min boundary
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_min, device=device)
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # y = y_max boundary
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_max, device=device)
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        coords_list.append((x, y, t))
        
        # Combine
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


# Register equations
from ..registry import register_equation
register_equation("heat", HeatEquation)
register_equation("heat_1d", HeatEquation1D)
register_equation("heat_2d", HeatEquation2D)


__all__ = [
    "HeatEquation",
    "HeatEquation1D",
    "HeatEquation2D",
]