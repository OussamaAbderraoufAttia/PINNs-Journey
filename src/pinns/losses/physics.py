"""
Loss Functions for PINNs

Implements physics loss, boundary loss, initial loss, and composite losses
with weighting strategies.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class LossType(Enum):
    """Types of loss components."""
    PHYSICS = "physics"
    BOUNDARY = "boundary"
    INITIAL = "initial"
    DATA = "data"
    REGULARIZATION = "regularization"


@dataclass
class LossWeights:
    """Weight configuration for composite loss."""
    physics: float = 1.0
    boundary: float = 1.0
    initial: float = 1.0
    data: float = 1.0
    regularization: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "physics": self.physics,
            "boundary": self.boundary,
            "initial": self.initial,
            "data": self.data,
            "regularization": self.regularization,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "LossWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LossComponent(ABC):
    """Abstract base class for loss components."""
    
    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """Compute loss component."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return loss component name."""
        pass


class PhysicsLoss(LossComponent):
    """
    Physics-informed loss (PDE/ODE residual).
    
    Computes: mean(|R(x)|²) where R is the equation residual.
    """
    
    def __init__(self, equation: Callable, weight: float = 1.0):
        self.equation = equation
        self.weight = weight
    
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        residual = self.equation.residual(model, coords)
        loss = torch.mean(residual**2)
        return self.weight * loss
    
    def name(self) -> str:
        return "physics"


class BoundaryLoss(LossComponent):
    """
    Boundary condition loss.
    
    Supports Dirichlet, Neumann, Robin, and Periodic BCs.
    """
    
    def __init__(
        self,
        equation: Callable,
        bc_type: str = "dirichlet",
        weight: float = 1.0,
    ):
        self.equation = equation
        self.bc_type = bc_type
        self.weight = weight
    
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        total_loss = torch.tensor(0.0, device=list(coords.values())[0].device)
        
        for bc in self.equation.boundary_conditions:
            # Filter points at this boundary
            mask = torch.ones_like(list(coords.values())[0], dtype=torch.bool)
            for dim, val in bc.location.items():
                mask = mask & (torch.abs(coords[dim] - val) < 1e-6)
            
            if not mask.any():
                continue
            
            # Get boundary coordinates
            bc_coords = {k: v[mask] for k, v in coords.items()}
            x = self.equation.domain.to_tensor(bc_coords)
            u_pred = model(x)
            
            if self.bc_type == "dirichlet":
                target = bc.value
                if callable(target):
                    target = target(**{k: v[mask] for k, v in bc_coords.items()})
                loss = torch.mean((u_pred - target)**2)
            elif self.bc_type == "neumann":
                # Normal derivative = value
                # Need to compute gradient normal to boundary
                normal = self._get_normal(bc.location)
                grad = self._compute_normal_derivative(model, bc_coords, normal)
                target = bc.value
                if callable(target):
                    target = target(**{k: v[mask] for k, v in bc_coords.items()})
                loss = torch.mean((grad - target)**2)
            elif self.bc_type == "periodic":
                # Periodic BC handled separately
                continue
            else:
                raise ValueError(f"Unknown BC type: {self.bc_type}")
            
            total_loss = total_loss + bc.weight * loss
        
        return self.weight * total_loss
    
    def _get_normal(self, location: Dict[str, float]) -> Dict[str, float]:
        """Get normal vector for boundary."""
        normal = {}
        for dim, val in location.items():
            if abs(val - 0) < 1e-6 or abs(val - 1) < 1e-6:
                normal[dim] = -1.0 if val == 0 else 1.0
        return normal
    
    def _compute_normal_derivative(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        normal: Dict[str, float],
    ) -> Tensor:
        """Compute directional derivative in normal direction."""
        x = self.equation.domain.to_tensor(coords)
        x.requires_grad_(True)
        u = model(x)
        
        grad = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        
        # Project onto normal
        normal_deriv = torch.zeros_like(u)
        for dim, n_val in normal.items():
            dim_idx = list(self.equation.domain.bounds.keys()).index(dim)
            normal_deriv = normal_deriv + n_val * grad[:, dim_idx:dim_idx+1]
        
        return normal_deriv
    
    def name(self) -> str:
        return "boundary"


class InitialLoss(LossComponent):
    """
    Initial condition loss for time-dependent problems.
    """
    
    def __init__(self, equation: Callable, weight: float = 1.0):
        self.equation = equation
        self.weight = weight
    
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        total_loss = torch.tensor(0.0, device=list(coords.values())[0].device)
        
        for ic in self.equation.initial_conditions:
            # Filter points at initial time
            mask = torch.ones_like(list(coords.values())[0], dtype=torch.bool)
            for dim, val in ic.location.items():
                mask = mask & (torch.abs(coords[dim] - val) < 1e-6)
            
            if not mask.any():
                continue
            
            ic_coords = {k: v[mask] for k, v in coords.items()}
            x = self.equation.domain.to_tensor(ic_coords)
            u_pred = model(x)
            
            target = ic.value
            if callable(target):
                target = target(**{k: v[mask] for k, v in ic_coords.items()})
            
            loss = torch.mean((u_pred - target)**2)
            total_loss = total_loss + ic.weight * loss
        
        return self.weight * total_loss
    
    def name(self) -> str:
        return "initial"


class DataLoss(LossComponent):
    """
    Supervised data loss for observed data points.
    """
    
    def __init__(
        self,
        data_coords: Dict[str, Tensor],
        data_values: Tensor,
        weight: float = 1.0,
    ):
        self.data_coords = data_coords
        self.data_values = data_values
        self.weight = weight
    
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        x = torch.cat([self.data_coords[k] for k in sorted(self.data_coords.keys())], dim=-1)
        u_pred = model(x)
        loss = torch.mean((u_pred - self.data_values)**2)
        return self.weight * loss
    
    def name(self) -> str:
        return "data"


class RegularizationLoss(LossComponent):
    """
    Regularization losses (L2, gradient penalty, etc.)
    """
    
    def __init__(
        self,
        model: nn.Module,
        reg_type: str = "l2",
        weight: float = 1e-4,
    ):
        self.model = model
        self.reg_type = reg_type
        self.weight = weight
    
    def compute(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        if self.reg_type == "l2":
            loss = sum(p.pow(2).sum() for p in model.parameters())
        elif self.reg_type == "l1":
            loss = sum(p.abs().sum() for p in model.parameters())
        elif self.reg_type == "gradient_penalty":
            # Gradient penalty on input
            x = torch.cat([coords[k] for k in sorted(coords.keys())], dim=-1)
            x.requires_grad_(True)
            u = model(x)
            grad = torch.autograd.grad(
                u.sum(), x, create_graph=True, retain_graph=True
            )[0]
            loss = torch.mean((grad.norm(dim=-1) - 1)**2)
        else:
            loss = torch.tensor(0.0)
        
        return self.weight * loss
    
    def name(self) -> str:
        return f"regularization_{self.reg_type}"


class AdaptiveLossWeighting:
    """
    Adaptive loss weighting strategies.
    
    Implements:
    - GradNorm (Chen et al., 2018)
    - NTK-based weighting (Wang et al., 2021)
    - Loss balancing (McClenny et al., 2023)
    """
    
    def __init__(
        self,
        strategy: str = "gradnorm",
        alpha: float = 1.5,
        update_freq: int = 100,
    ):
        self.strategy = strategy
        self.alpha = alpha
        self.update_freq = update_freq
        self.step_count = 0
        self.initial_losses = None
        self.loss_weights = None
    
    def update(self, losses: Dict[str, Tensor], model: nn.Module) -> Dict[str, float]:
        """Update loss weights based on strategy."""
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            return self.loss_weights or {k: 1.0 for k in losses.keys()}
        
        loss_values = {k: v.item() for k, v in losses.items()}
        
        if self.initial_losses is None:
            self.initial_losses = loss_values.copy()
            self.loss_weights = {k: 1.0 for k in losses.keys()}
            return self.loss_weights
        
        if self.strategy == "gradnorm":
            self.loss_weights = self._gradnorm_weighting(loss_values, model)
        elif self.strategy == "ntk":
            self.loss_weights = self._ntk_weighting(loss_values)
        elif self.strategy == "loss_balancing":
            self.loss_weights = self._loss_balancing(loss_values)
        
        return self.loss_weights
    
    def _gradnorm_weighting(
        self,
        losses: Dict[str, float],
        model: nn.Module,
    ) -> Dict[str, float]:
        """GradNorm: balance gradient magnitudes."""
        # Compute gradient norms for each loss
        grad_norms = {}
        for name, loss_val in losses.items():
            # This is simplified; full GradNorm computes grad w.r.t. shared params
            grad_norms[name] = loss_val  # Placeholder
        
        # Target gradient norm (average)
        target_norm = sum(grad_norms.values()) / len(grad_norms)
        
        # Update weights
        new_weights = {}
        for name in losses:
            ratio = grad_norms[name] / (target_norm + 1e-8)
            new_weights[name] = ratio ** self.alpha
        
        # Normalize
        total = sum(new_weights.values())
        return {k: v / total * len(new_weights) for k, v in new_weights.items()}
    
    def _ntk_weighting(self, losses: Dict[str, float]) -> Dict[str, float]:
        """NTK-based weighting: weight by inverse of initial loss."""
        new_weights = {}
        for name, loss_val in losses.items():
            initial = self.initial_losses.get(name, loss_val)
            new_weights[name] = initial / (loss_val + 1e-8)
        
        # Normalize
        total = sum(new_weights.values())
        return {k: v / total * len(new_weights) for k, v in new_weights.items()}
    
    def _loss_balancing(self, losses: Dict[str, float]) -> Dict[str, float]:
        """Simple loss balancing: weight inversely proportional to loss."""
        # Moving average of losses
        if not hasattr(self, 'loss_ema'):
            self.loss_ema = {k: v for k, v in losses.items()}
        else:
            for k, v in losses.items():
                self.loss_ema[k] = 0.9 * self.loss_ema[k] + 0.1 * v
        
        # Weight inversely proportional to EMA
        new_weights = {}
        for name, ema in self.loss_ema.items():
            new_weights[name] = 1.0 / (ema + 1e-8)
        
        total = sum(new_weights.values())
        return {k: v / total * len(new_weights) for k, v in new_weights.items()}


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss components.
    
    Supports:
    - Fixed weights
    - Adaptive weighting (GradNorm, NTK, Loss Balancing)
    - Loss logging
    """
    
    def __init__(
        self,
        loss_components: List[LossComponent],
        weights: Optional[LossWeights] = None,
        adaptive_weighting: Optional[AdaptiveLossWeighting] = None,
    ):
        super().__init__()
        self.loss_components = nn.ModuleList(loss_components)
        self.weights = weights or LossWeights()
        self.adaptive_weighting = adaptive_weighting
        self.loss_history = {comp.name(): [] for comp in loss_components}
    
    def forward(
        self,
        model: nn.Module,
        coords: Dict[str, Tensor],
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components.
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        for component in self.loss_components:
            name = component.name()
            loss = component.compute(model, coords, **kwargs)
            losses[name] = loss
            self.loss_history[name].append(loss.item())
        
        # Apply adaptive weighting if enabled
        if self.adaptive_weighting is not None:
            adaptive_weights = self.adaptive_weighting.update(losses, model)
            for name in losses:
                if name in adaptive_weights:
                    losses[name] = losses[name] * adaptive_weights[name]
        
        # Apply fixed weights
        weight_dict = self.weights.to_dict()
        for name in losses:
            if name in weight_dict:
                losses[name] = losses[name] * weight_dict[name]
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        return self.loss_history
    
    def reset_history(self):
        self.loss_history = {comp.name(): [] for comp in self.loss_components}


def create_pinn_loss(
    equation: Callable,
    weights: Optional[LossWeights] = None,
    adaptive: bool = False,
    adaptive_strategy: str = "gradnorm",
) -> CompositeLoss:
    """
    Factory function to create standard PINN loss.
    
    Args:
        equation: Equation object with residual, BCs, ICs
        weights: LossWeights object
        adaptive: Whether to use adaptive weighting
        adaptive_strategy: "gradnorm", "ntk", or "loss_balancing"
    
    Returns:
        CompositeLoss instance
    """
    components = [
        PhysicsLoss(equation),
        BoundaryLoss(equation),
        InitialLoss(equation),
    ]
    
    if adaptive:
        adaptive_weighting = AdaptiveLossWeighting(strategy=adaptive_strategy)
    else:
        adaptive_weighting = None
    
    return CompositeLoss(components, weights, adaptive_weighting)


__all__ = [
    "LossType",
    "LossWeights",
    "LossComponent",
    "PhysicsLoss",
    "BoundaryLoss",
    "InitialLoss",
    "DataLoss",
    "RegularizationLoss",
    "AdaptiveLossWeighting",
    "CompositeLoss",
    "create_pinn_loss",
]