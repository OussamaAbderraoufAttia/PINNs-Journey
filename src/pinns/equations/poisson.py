"""
Poisson Equation Implementation

-∇²u = f  (or ∇²u = f)

1D: -u_xx = f(x)
2D: -(u_xx + u_yy) = f(x,y)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Callable
import math

from .base import PDE, Domain, BoundaryCondition, InitialCondition


class PoissonEquation(PDE):
    """
    General Poisson equation: -∇²u = f
    """
    
    def __init__(
        self,
        domain: Domain,
        source_term: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        parameters: Dict = None,
    ):
        parameters = parameters or {}
        super().__init__(domain, boundary_conditions, [], parameters)
        self.source_term = source_term or (lambda **kwargs: torch.zeros_like(list(kwargs.values())[0]))
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Use PoissonEquation1D or PoissonEquation2D")
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        return None


class PoissonEquation1D(PoissonEquation):
    """
    1D Poisson Equation: -u_xx = f(x)
    
    Domain: x ∈ [0, 1]
    
    Example with analytical solution:
    u(x) = sin(πx)  ->  f(x) = π² sin(πx)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        source_term: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        analytical_solution: Callable = None,
    ):
        domain = Domain(bounds={"x": (x_min, x_max)})
        
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
            ]
        
        super().__init__(domain, source_term, boundary_conditions, {})
        self.source_term = source_term or (lambda x: math.pi**2 * torch.sin(math.pi * x))
        self.analytical_solution_fn = analytical_solution or (lambda x: torch.sin(math.pi * x))
        self.x_min, self.x_max = x_min, x_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute Poisson residual: -u_xx - f(x) = 0"""
        x = coords['x']
        u = model(x)
        
        # Second derivative
        u_x = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), x, create_graph=True, retain_graph=True
        )[0]
        
        # Residual: -u_xx - f(x)
        f = self.source_term(x)
        residual = -u_xx - f
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        return self.analytical_solution_fn(coords['x'])
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n = n_per_boundary
        x_left = torch.full((n, 1), self.x_min, device=device)
        x_right = torch.full((n, 1), self.x_max, device=device)
        
        x = torch.cat([x_left, x_right], dim=0).requires_grad_(True)
        return {"x": x}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Not applicable for steady-state problems."""
        raise NotImplementedError("Poisson equation is steady-state; use sample_domain")


class PoissonEquation2D(PoissonEquation):
    """
    2D Poisson Equation: -(u_xx + u_yy) = f(x,y)
    
    Domain: x ∈ [0, 1], y ∈ [0, 1]
    
    Example with analytical solution:
    u(x,y) = sin(πx)sin(πy)  ->  f(x,y) = 2π² sin(πx)sin(πy)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        source_term: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        analytical_solution: Callable = None,
    ):
        domain = Domain(bounds={"x": (x_min, x_max), "y": (y_min, y_max)})
        
        if boundary_conditions is None:
            boundary_conditions = [
                BoundaryCondition("dirichlet", {"x": x_min}, 0.0),
                BoundaryCondition("dirichlet", {"x": x_max}, 0.0),
                BoundaryCondition("dirichlet", {"y": y_min}, 0.0),
                BoundaryCondition("dirichlet", {"y": y_max}, 0.0),
            ]
        
        super().__init__(domain, source_term, boundary_conditions, {})
        self.source_term = source_term or (lambda x, y: 2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y))
        self.analytical_solution_fn = analytical_solution or (lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y))
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute 2D Poisson residual: -(u_xx + u_yy) - f = 0"""
        x = torch.cat([coords['x'], coords['y']], dim=-1)
        u = model(x)
        
        laplacian = 0
        for dim in ['x', 'y']:
            u_d = torch.autograd.grad(
                u.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            u_dd = torch.autograd.grad(
                u_d.sum(), coords[dim], create_graph=True, retain_graph=True
            )[0]
            laplacian = laplacian + u_dd
        
        f = self.source_term(coords['x'], coords['y'])
        residual = -laplacian - f
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Tensor:
        return self.analytical_solution_fn(coords['x'], coords['y'])
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n = n_per_boundary
        coords_list = []
        
        # x = x_min
        x = torch.full((n, 1), self.x_min, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        coords_list.append((x, y))
        
        # x = x_max
        x = torch.full((n, 1), self.x_max, device=device)
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        coords_list.append((x, y))
        
        # y = y_min
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_min, device=device)
        coords_list.append((x, y))
        
        # y = y_max
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.full((n, 1), self.y_max, device=device)
        coords_list.append((x, y))
        
        x_all = torch.cat([c[0] for c in coords_list], dim=0).requires_grad_(True)
        y_all = torch.cat([c[1] for c in coords_list], dim=0).requires_grad_(True)
        
        return {"x": x_all, "y": y_all}
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        raise NotImplementedError("Poisson equation is steady-state; use sample_domain")


from ..registry import register_equation
register_equation("poisson", PoissonEquation)
register_equation("poisson_1d", PoissonEquation1D)
register_equation("poisson_2d", PoissonEquation2D)


__all__ = [
    "PoissonEquation",
    "PoissonEquation1D",
    "PoissonEquation2D",
]