"""
Reaction-Diffusion Equation Implementation

u_t = D ∇²u + f(u)

Common forms:
- Fisher-KPP: u_t = D u_xx + r u (1 - u)
- Allen-Cahn: u_t = D u_xx + u - u³
- Gray-Scott: u_t = D₁ u_xx - u v² + F(1-u)
              v_t = D₂ v_xx + u v² - (F+k)v
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Callable
import math

from .base import PDE, Domain, BoundaryCondition, InitialCondition


class ReactionDiffusionEquation(PDE):
    """
    General reaction-diffusion equation: u_t = D ∇²u + f(u)
    """
    
    def __init__(
        self,
        domain: Domain,
        diffusion_coeff: float = 0.01,
        reaction_term: Callable = None,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
    ):
        parameters = {"D": diffusion_coeff}
        super().__init__(domain, boundary_conditions, initial_conditions, parameters)
        self.D = diffusion_coeff
        self.reaction_term = reaction_term or (lambda u: u * (1 - u))  # Fisher-KPP
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Use ReactionDiffusion1D or 2D")


class ReactionDiffusion1D(ReactionDiffusionEquation):
    """
    1D Reaction-Diffusion: u_t = D u_xx + f(u)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        D: float = 0.01,
        reaction_type: str = "fisher_kpp",
        r: float = 1.0,
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
                InitialCondition({"t": t_min}, lambda x: torch.exp(-100 * (x - 0.5)**2))
            ]
        
        # Define reaction term
        if reaction_type == "fisher_kpp":
            reaction = lambda u: r * u * (1 - u)
        elif reaction_type == "allen_cahn":
            reaction = lambda u: u - u**3
        elif reaction_type == "bistable":
            reaction = lambda u: u * (1 - u) * (u - 0.5)
        else:
            reaction = lambda u: u * (1 - u)
        
        super().__init__(domain, D, reaction, boundary_conditions, initial_conditions)
        self.reaction_type = reaction_type
        self.r = r
        self.x_min, self.x_max = x_min, x_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute reaction-diffusion residual: u_t - D u_xx - f(u) = 0"""
        x = torch.cat([coords['x'], coords['t']], dim=-1)
        u = model(x)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
        )[0]
        
        # Spatial derivatives
        u_x = torch.autograd.grad(
            u.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), coords['x'], create_graph=True, retain_graph=True
        )[0]
        
        # Reaction term
        f_u = self.reaction_term(u)
        
        # Residual
        residual = u_t - self.D * u_xx - f_u
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        """No general analytical solution."""
        return None
    
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


class ReactionDiffusion2D(ReactionDiffusionEquation):
    """
    2D Reaction-Diffusion: u_t = D (u_xx + u_yy) + f(u)
    """
    
    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        D: float = 0.01,
        reaction_type: str = "fisher_kpp",
        r: float = 1.0,
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
                    lambda x, y: torch.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))
                )
            ]
        
        if reaction_type == "fisher_kpp":
            reaction = lambda u: r * u * (1 - u)
        elif reaction_type == "allen_cahn":
            reaction = lambda u: u - u**3
        else:
            reaction = lambda u: u * (1 - u)
        
        super().__init__(domain, D, reaction, boundary_conditions, initial_conditions)
        self.reaction_type = reaction_type
        self.r = r
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
    
    def residual(self, model: callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute 2D reaction-diffusion residual."""
        x = torch.cat([coords['x'], coords['y'], coords['t']], dim=-1)
        u = model(x)
        
        # Time derivative
        u_t = torch.autograd.grad(
            u.sum(), coords['t'], create_graph=True, retain_graph=True
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
        
        f_u = self.reaction_term(u)
        
        residual = u_t - self.D * laplacian - f_u
        return residual
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        return None
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
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
    "ReactionDiffusionEquation",
    "ReactionDiffusion1D",
    "ReactionDiffusion2D",
]