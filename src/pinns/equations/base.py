"""
Base Equation Classes

Abstract base classes for ODEs and PDEs with common interface
for residual computation, boundary conditions, and analytical solutions.
"""

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Domain:
    """Computational domain definition."""
    bounds: Dict[str, Tuple[float, float]]  # e.g., {"x": (0, 1), "t": (0, 1)}
    
    def sample_uniform(self, n: int, requires_grad: bool = True) -> Dict[str, Tensor]:
        """Sample n points uniformly from domain."""
        samples = {}
        for dim, (low, high) in self.bounds.items():
            samples[dim] = torch.rand(n, 1) * (high - low) + low
            if requires_grad:
                samples[dim].requires_grad_(True)
        return samples
    
    def sample_latin_hypercube(self, n: int, requires_grad: bool = True) -> Dict[str, Tensor]:
        """Sample using Latin Hypercube Sampling (better space-filling)."""
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(self.bounds))
        samples = sampler.random(n)
        
        result = {}
        for i, (dim, (low, high)) in enumerate(self.bounds.items()):
            result[dim] = torch.tensor(samples[:, i:i+1]) * (high - low) + low
            if requires_grad:
                result[dim].requires_grad_(True)
        return result
    
    def sample_sobol(self, n: int, requires_grad: bool = True) -> Dict[str, Tensor]:
        """Sample using Sobol sequence (low-discrepancy)."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=len(self.bounds), scramble=True)
        samples = sampler.random(n)
        
        result = {}
        for i, (dim, (low, high)) in enumerate(self.bounds.items()):
            result[dim] = torch.tensor(samples[:, i:i+1]) * (high - low) + low
            if requires_grad:
                result[dim].requires_grad_(True)
        return result
    
    def to_tensor(self, samples: Dict[str, Tensor]) -> Tensor:
        """Convert dictionary of samples to concatenated tensor."""
        # Sort keys for consistent ordering
        sorted_keys = sorted(samples.keys())
        return torch.cat([samples[k] for k in sorted_keys], dim=-1)
    
    def from_tensor(self, tensor: Tensor) -> Dict[str, Tensor]:
        """Convert concatenated tensor back to dictionary."""
        sorted_keys = sorted(self.bounds.keys())
        result = {}
        for i, key in enumerate(sorted_keys):
            result[key] = tensor[:, i:i+1]
        return result


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    bc_type: str  # "dirichlet", "neumann", "robin", "periodic"
    location: Dict[str, float]  # e.g., {"x": 0.0} for left boundary
    value: Any  # Constant or callable
    weight: float = 1.0
    
    def apply(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        """Apply boundary condition and return residual."""
        # Filter points at this boundary
        mask = torch.ones_like(list(coords.values())[0], dtype=torch.bool)
        for dim, val in self.location.items():
            mask = mask & (torch.abs(coords[dim] - val) < 1e-6)
        
        if not mask.any():
            return torch.tensor(0.0, device=list(coords.values())[0].device)
        
        # Get model prediction at boundary points
        boundary_coords = {k: v[mask] for k, v in coords.items()}
        u_pred = model(self.domain.to_tensor(boundary_coords))
        
        if self.bc_type == "dirichlet":
            target = self.value
            if callable(target):
                target = target(**{k: v[mask] for k, v in boundary_coords.items()})
            return u_pred - target
        elif self.bc_type == "neumann":
            # Normal derivative = value
            # Need to compute gradient
            raise NotImplementedError("Neumann BC requires gradient computation")
        else:
            raise ValueError(f"Unknown BC type: {self.bc_type}")


@dataclass
class InitialCondition:
    """Initial condition specification."""
    location: Dict[str, float]  # e.g., {"t": 0.0}
    value: Any  # Constant or callable
    weight: float = 1.0


class Equation(ABC):
    """
    Abstract base class for all equations.
    
    Provides interface for:
    - Residual computation (physics loss)
    - Boundary/initial condition residuals
    - Analytical solution (if available)
    - Domain sampling
    """
    
    def __init__(
        self,
        domain: Domain,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
        parameters: Dict[str, float] = None,
    ):
        self.domain = domain
        self.boundary_conditions = boundary_conditions or []
        self.initial_conditions = initial_conditions or []
        self.parameters = parameters or {}
    
    @abstractmethod
    def residual(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        """
        Compute PDE/ODE residual at given coordinates.
        
        Args:
            model: Neural network model (callable)
            coords: Dictionary of coordinate tensors with requires_grad=True
        
        Returns:
            Residual tensor of shape (N,)
        """
        pass
    
    def boundary_residual(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute boundary condition residuals."""
        total_residual = torch.tensor(0.0, device=list(coords.values())[0].device)
        for bc in self.boundary_conditions:
            res = bc.apply(model, coords)
            total_residual = total_residual + bc.weight * res.pow(2).mean()
        return total_residual
    
    def initial_residual(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        """Compute initial condition residuals."""
        total_residual = torch.tensor(0.0, device=list(coords.values())[0].device)
        for ic in self.initial_conditions:
            # Filter points at initial time
            mask = torch.ones_like(list(coords.values())[0], dtype=torch.bool)
            for dim, val in ic.location.items():
                mask = mask & (torch.abs(coords[dim] - val) < 1e-6)
            
            if not mask.any():
                continue
            
            ic_coords = {k: v[mask] for k, v in coords.items()}
            u_pred = model(self.domain.to_tensor(ic_coords))
            
            target = ic.value
            if callable(target):
                target = target(**{k: v[mask] for k, v in ic_coords.items()})
            
            total_residual = total_residual + ic.weight * (u_pred - target).pow(2).mean()
        
        return total_residual
    
    @abstractmethod
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        """
        Analytical solution if available.
        
        Returns:
            Solution tensor or None if not available
        """
        pass
    
    def sample_domain(self, n: int, strategy: str = "uniform") -> Dict[str, Tensor]:
        """Sample points from domain."""
        if strategy == "uniform":
            return self.domain.sample_uniform(n)
        elif strategy == "lhs":
            return self.domain.sample_latin_hypercube(n)
        elif strategy == "sobol":
            return self.domain.sample_sobol(n)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def sample_boundary(self, n_per_boundary: int) -> Dict[str, Tensor]:
        """Sample points on boundaries."""
        # This is equation-specific, implemented in subclasses
        raise NotImplementedError
    
    def sample_initial(self, n: int) -> Dict[str, Tensor]:
        """Sample points at initial time."""
        # This is equation-specific, implemented in subclasses
        raise NotImplementedError


class ODE(Equation):
    """Base class for Ordinary Differential Equations."""
    
    def __init__(
        self,
        domain: Domain,
        initial_conditions: List[InitialCondition] = None,
        parameters: Dict[str, float] = None,
    ):
        # ODEs typically don't have boundary conditions
        super().__init__(domain, boundary_conditions=[], initial_conditions=initial_conditions, parameters=parameters)
    
    @abstractmethod
    def residual(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        pass
    
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        return None


class PDE(Equation):
    """Base class for Partial Differential Equations."""
    
    def __init__(
        self,
        domain: Domain,
        boundary_conditions: List[BoundaryCondition] = None,
        initial_conditions: List[InitialCondition] = None,
        parameters: Dict[str, float] = None,
    ):
        super().__init__(domain, boundary_conditions, initial_conditions, parameters)
    
    @abstractmethod
    def residual(self, model: Callable, coords: Dict[str, Tensor]) -> Tensor:
        pass
    
    @abstractmethod
    def analytical_solution(self, coords: Dict[str, Tensor]) -> Optional[Tensor]:
        pass


def compute_residual_ode(
    model: Callable,
    t: Tensor,
    ode_func: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """
    Compute ODE residual: du/dt - f(t, u) = 0.
    
    Args:
        model: Neural network u(t)
        t: Time coordinates (N, 1) with requires_grad=True
        ode_func: Function f(t, u) defining the ODE
    
    Returns:
        Residual tensor (N,)
    """
    u = model(t)
    du_dt = torch.autograd.grad(
        u.sum(), t, create_graph=True, retain_graph=True
    )[0]
    f = ode_func(t, u)
    return du_dt - f


def compute_residual_pde(
    model: Callable,
    coords: Dict[str, Tensor],
    pde_func: Callable[..., Tensor],
    coord_names: List[str],
) -> Tensor:
    """
    Compute PDE residual using automatic differentiation.
    
    Args:
        model: Neural network
        coords: Dictionary of coordinate tensors with requires_grad=True
        pde_func: Function computing PDE residual from u and its derivatives
        coord_names: Ordered list of coordinate names
    
    Returns:
        Residual tensor
    """
    # Concatenate coordinates for model input
    x = torch.cat([coords[name] for name in coord_names], dim=-1)
    u = model(x)
    
    # Compute derivatives needed by PDE
    # This is a generic framework; specific PDEs implement their own
    return pde_func(u, coords)


__all__ = [
    "Domain",
    "BoundaryCondition",
    "InitialCondition",
    "Equation",
    "ODE",
    "PDE",
    "compute_residual_ode",
    "compute_residual_pde",
]