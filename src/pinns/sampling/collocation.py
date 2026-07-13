"""
Sampling Strategies for PINNs

Implements various collocation point sampling strategies:
- Uniform random
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Adaptive sampling (RAR, RARD, etc.)
- Importance sampling based on residuals
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.stats import qmc


@dataclass
class SamplingConfig:
    """Configuration for sampling strategy."""
    strategy: str = "uniform"  # uniform, lhs, sobol, adaptive, residual
    n_collocation: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000
    n_data: int = 0
    adaptive_k: int = 10  # Top-k points for adaptive sampling
    adaptive_freq: int = 1000  # Update frequency
    residual_weight: float = 1.0


class Sampler(ABC):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        """Sample n points from domain."""
        pass
    
    @abstractmethod
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        """Sample points on boundaries."""
        pass
    
    @abstractmethod
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        """Sample initial condition points."""
        pass


class UniformSampler(Sampler):
    """Uniform random sampling."""
    
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        samples = {}
        for dim, (low, high) in bounds.items():
            samples[dim] = torch.rand(n, 1) * (high - low) + low
            samples[dim].requires_grad_(True)
        return samples
    
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        samples = {dim: [] for dim in bounds}
        
        for bc in boundary_conditions:
            loc = bc.get("location", {})
            for dim, (low, high) in bounds.items():
                if dim in loc:
                    # This dimension is fixed at boundary
                    samples[dim].append(torch.full((n_per_boundary, 1), loc[dim]))
                else:
                    # Other dimensions vary uniformly
                    samples[dim].append(torch.rand(n_per_boundary, 1) * (high - low) + low)
        
        result = {}
        for dim in bounds:
            result[dim] = torch.cat(samples[dim], dim=0).requires_grad_(True)
        return result
    
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        # Find time dimension
        time_dim = None
        for dim, (low, high) in bounds.items():
            if dim in ['t', 'time']:
                time_dim = dim
                break
        
        if time_dim is None:
            raise ValueError("No time dimension found in bounds")
        
        samples = {}
        for dim, (low, high) in bounds.items():
            if dim == time_dim:
                # Find initial time value
                t_init = initial_conditions[0].get("location", {}).get(dim, low)
                samples[dim] = torch.full((n, 1), t_init)
            else:
                samples[dim] = torch.rand(n, 1) * (high - low) + low
            samples[dim].requires_grad_(True)
        return samples


class LHSSampler(Sampler):
    """Latin Hypercube Sampling for better space-filling."""
    
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        dims = len(bounds)
        sampler = qmc.LatinHypercube(d=dims, seed=np.random.randint(0, 2**32))
        unit_samples = sampler.random(n)
        
        result = {}
        for i, (dim, (low, high)) in enumerate(bounds.items()):
            result[dim] = torch.tensor(
                unit_samples[:, i:i+1] * (high - low) + low,
                dtype=torch.float32
            ).requires_grad_(True)
        return result
    
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        # For boundaries, use LHS on the free dimensions
        samples = {dim: [] for dim in bounds}
        
        for bc in boundary_conditions:
            loc = bc.get("location", {})
            fixed_dims = list(loc.keys())
            free_dims = [d for d in bounds if d not in fixed_dims]
            
            if free_dims:
                sampler = qmc.LatinHypercube(d=len(free_dims))
                unit_samples = sampler.random(n_per_boundary)
                
                for i, dim in enumerate(free_dims):
                    low, high = bounds[dim]
                    samples[dim].append(torch.tensor(
                        unit_samples[:, i:i+1] * (high - low) + low,
                        dtype=torch.float32
                    ))
            else:
                for dim in bounds:
                    samples[dim].append(torch.full((n_per_boundary, 1), loc.get(dim, bounds[dim][0])))
            
            # Add fixed dimensions
            for dim, val in loc.items():
                samples[dim].append(torch.full((n_per_boundary, 1), val))
        
        result = {}
        for dim in bounds:
            result[dim] = torch.cat(samples[dim], dim=0).requires_grad_(True)
        return result
    
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        time_dim = None
        for dim in bounds:
            if dim in ['t', 'time']:
                time_dim = dim
                break
        
        if time_dim is None:
            raise ValueError("No time dimension found")
        
        spatial_dims = [d for d in bounds if d != time_dim]
        
        if spatial_dims:
            sampler = qmc.LatinHypercube(d=len(spatial_dims))
            unit_samples = sampler.random(n)
            
            result = {}
            for i, dim in enumerate(spatial_dims):
                low, high = bounds[dim]
                result[dim] = torch.tensor(
                    unit_samples[:, i:i+1] * (high - low) + low,
                    dtype=torch.float32
                ).requires_grad_(True)
            
            t_init = initial_conditions[0].get("location", {}).get(time_dim, bounds[time_dim][0])
            result[time_dim] = torch.full((n, 1), t_init).requires_grad_(True)
            return result
        else:
            t_init = initial_conditions[0].get("location", {}).get(time_dim, bounds[time_dim][0])
            return {time_dim: torch.full((n, 1), t_init).requires_grad_(True)}


class SobolSampler(Sampler):
    """Sobol sequence sampling (low-discrepancy)."""
    
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        dims = len(bounds)
        sampler = qmc.Sobol(d=dims, scramble=True)
        unit_samples = sampler.random(n)
        
        result = {}
        for i, (dim, (low, high)) in enumerate(bounds.items()):
            result[dim] = torch.tensor(
                unit_samples[:, i:i+1] * (high - low) + low,
                dtype=torch.float32
            ).requires_grad_(True)
        return result
    
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        # Use Sobol on free dimensions
        samples = {dim: [] for dim in bounds}
        
        for bc in boundary_conditions:
            loc = bc.get("location", {})
            fixed_dims = list(loc.keys())
            free_dims = [d for d in bounds if d not in fixed_dims]
            
            if free_dims:
                sampler = qmc.Sobol(d=len(free_dims), scramble=True)
                unit_samples = sampler.random(n_per_boundary)
                
                for i, dim in enumerate(free_dims):
                    low, high = bounds[dim]
                    samples[dim].append(torch.tensor(
                        unit_samples[:, i:i+1] * (high - low) + low,
                        dtype=torch.float32
                    ))
            else:
                for dim in bounds:
                    samples[dim].append(torch.full((n_per_boundary, 1), loc.get(dim, bounds[dim][0])))
            
            for dim, val in loc.items():
                samples[dim].append(torch.full((n_per_boundary, 1), val))
        
        result = {}
        for dim in bounds:
            result[dim] = torch.cat(samples[dim], dim=0).requires_grad_(True)
        return result
    
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        time_dim = None
        for dim in bounds:
            if dim in ['t', 'time']:
                time_dim = dim
                break
        
        spatial_dims = [d for d in bounds if d != time_dim]
        
        if spatial_dims:
            sampler = qmc.Sobol(d=len(spatial_dims), scramble=True)
            unit_samples = sampler.random(n)
            
            result = {}
            for i, dim in enumerate(spatial_dims):
                low, high = bounds[dim]
                result[dim] = torch.tensor(
                    unit_samples[:, i:i+1] * (high - low) + low,
                    dtype=torch.float32
                ).requires_grad_(True)
            
            t_init = initial_conditions[0].get("location", {}).get(time_dim, bounds[time_dim][0])
            result[time_dim] = torch.full((n, 1), t_init).requires_grad_(True)
            return result
        else:
            t_init = initial_conditions[0].get("location", {}).get(time_dim, bounds[time_dim][0])
            return {time_dim: torch.full((n, 1), t_init).requires_grad_(True)}


class AdaptiveSampler(Sampler):
    """
    Adaptive sampling based on residual magnitude (RAR, RARD, etc.)
    
    Strategies:
    - RAR: Residual-based Adaptive Refinement (add points where residual is large)
    - RARD: RAR with Distribution (also consider point density)
    - Importance: Sample proportionally to residual
    """
    
    def __init__(
        self,
        base_sampler: Sampler = None,
        strategy: str = "rar",
        top_k: int = 10,
        update_freq: int = 1000,
    ):
        self.base_sampler = base_sampler or UniformSampler()
        self.strategy = strategy
        self.top_k = top_k
        self.update_freq = update_freq
        self.step_count = 0
        self.candidate_points = None
        self.residuals = None
    
    def update_residuals(self, model: Callable, equation: Callable, coords: Dict[str, Tensor]):
        """Update residuals for adaptive sampling."""
        self.step_count += 1
        
        if self.step_count % self.update_freq != 0:
            return
        
        with torch.no_grad():
            x = equation.domain.to_tensor(coords)
            residual = equation.residual(model, coords)
            self.residuals = residual.detach()
        
        # Generate new candidate points
        n_candidates = len(coords[list(coords.keys())[0]]) * 5
        self.candidate_points = self.base_sampler.sample_domain(
            n_candidates, equation.domain.bounds
        )
        
        # Evaluate residuals at candidates
        with torch.no_grad():
            x_cand = equation.domain.to_tensor(self.candidate_points)
            cand_residual = equation.residual(model, self.candidate_points)
            self.candidate_residuals = cand_residual.detach()
    
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        if self.candidate_points is not None and self.candidate_residuals is not None:
            # Select top-k candidates based on strategy
            if self.strategy == "rar":
                # Pure residual-based
                _, indices = torch.topk(self.candidate_residuals.flatten(), min(self.top_k, n))
            elif self.strategy == "importance":
                # Importance sampling proportional to residual
                probs = self.candidate_residuals.flatten() / (self.candidate_residuals.sum() + 1e-8)
                indices = torch.multinomial(probs, min(n, len(probs)), replacement=False)
            else:
                indices = torch.randperm(len(self.candidate_residuals))[:n]
            
            # Mix with uniform sampling
            n_adaptive = min(self.top_k, n // 2)
            n_uniform = n - n_adaptive
            
            adaptive_samples = {k: v[indices[:n_adaptive]] for k, v in self.candidate_points.items()}
            uniform_samples = self.base_sampler.sample_domain(n_uniform, bounds)
            
            result = {}
            for k in bounds:
                result[k] = torch.cat([adaptive_samples[k], uniform_samples[k]], dim=0)
            return result
        else:
            return self.base_sampler.sample_domain(n, bounds)
    
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        return self.base_sampler.sample_boundary(n_per_boundary, bounds, boundary_conditions)
    
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        return self.base_sampler.sample_initial(n, bounds, initial_conditions)


class ResidualBasedSampler(Sampler):
    """
    Sample points proportional to residual magnitude.
    
    Reference: "Adaptive Sampling for Physics-Informed Neural Networks" (Wu et al.)
    """
    
    def __init__(
        self,
        model: Callable,
        equation: Callable,
        bounds: Dict[str, Tuple[float, float]],
        n_candidates: int = 10000,
        base_sampler: Sampler = None,
    ):
        self.model = model
        self.equation = equation
        self.bounds = bounds
        self.n_candidates = n_candidates
        self.base_sampler = base_sampler or UniformSampler()
        self.sampling_distribution = None
    
    def update_distribution(self):
        """Update sampling distribution based on current residuals."""
        # Generate candidate points
        candidates = self.base_sampler.sample_domain(self.n_candidates, self.bounds)
        
        with torch.no_grad():
            residuals = self.equation.residual(self.model, candidates)
            residuals = residuals.flatten().abs()
        
        # Normalize to probability distribution
        self.sampling_distribution = residuals / (residuals.sum() + 1e-8)
        self.candidates = candidates
    
    def sample_domain(self, n: int, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tensor]:
        if self.sampling_distribution is None:
            self.update_distribution()
        
        # Sample from distribution
        indices = torch.multinomial(self.sampling_distribution, n, replacement=True)
        
        result = {}
        for dim in bounds:
            result[dim] = self.candidates[dim][indices].requires_grad_(True)
        return result
    
    def sample_boundary(
        self,
        n_per_boundary: int,
        bounds: Dict[str, Tuple[float, float]],
        boundary_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        return self.base_sampler.sample_boundary(n_per_boundary, bounds, boundary_conditions)
    
    def sample_initial(
        self,
        n: int,
        bounds: Dict[str, Tuple[float, float]],
        initial_conditions: List[Dict],
    ) -> Dict[str, Tensor]:
        return self.base_sampler.sample_initial(n, bounds, initial_conditions)


def create_sampler(config: SamplingConfig) -> Sampler:
    """Factory function to create sampler from config."""
    if config.strategy == "uniform":
        return UniformSampler()
    elif config.strategy == "lhs":
        return LHSSampler()
    elif config.strategy == "sobol":
        return SobolSampler()
    elif config.strategy == "adaptive":
        return AdaptiveSampler(
            strategy="rar",
            top_k=config.adaptive_k,
            update_freq=config.adaptive_freq,
        )
    elif config.strategy == "residual":
        # Requires model and equation - created later
        return None
    else:
        raise ValueError(f"Unknown sampling strategy: {config.strategy}")


__all__ = [
    "SamplingConfig",
    "Sampler",
    "UniformSampler",
    "LHSSampler",
    "SobolSampler",
    "AdaptiveSampler",
    "ResidualBasedSampler",
    "create_sampler",
]