"""
Activation Functions

Collection of activation functions for PINNs with their derivatives
and properties relevant to scientific computing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinActivation(nn.Module):
    """
    Sine activation: sin(x)
    
    Properties:
    - Periodic, bounded [-1, 1]
    - Derivatives: cos(x), -sin(x), -cos(x), sin(x), ...
    - Good for periodic/high-frequency functions
    - Spectral bias: learns low frequencies first
    """
    
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)
    
    def derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute analytical derivative."""
        w0 = self.w0
        if order == 1:
            return w0 * torch.cos(w0 * x)
        elif order == 2:
            return -w0**2 * torch.sin(w0 * x)
        elif order == 3:
            return -w0**3 * torch.cos(w0 * x)
        elif order == 4:
            return w0**4 * torch.sin(w0 * x)
        else:
            # General pattern: sin^(n)(x) = sin(x + n*pi/2)
            phase = order * torch.pi / 2
            return w0**order * torch.sin(w0 * x + phase)


class SwishActivation(nn.Module):
    """
    Swish activation: x * sigmoid(beta * x)
    
    Properties:
    - Smooth, non-monotonic
    - Self-gated
    - Beta=1: SiLU (used in transformers)
    - Beta->inf: ReLU
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)
    
    def derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        if order == 1:
            sigmoid = torch.sigmoid(self.beta * x)
            return sigmoid + self.beta * x * sigmoid * (1 - sigmoid)
        else:
            # Use autograd for higher orders
            x.requires_grad_(True)
            y = self.forward(x)
            for _ in range(order):
                y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            return y


class GaussianActivation(nn.Module):
    """
    Gaussian activation: exp(-x^2)
    
    Properties:
    - Bounded [0, 1]
    - Smooth, decays rapidly
    - Good for localized features
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x**2)


class RationalActivation(nn.Module):
    """
    Rational activation: P(x) / Q(x)
    
    Can approximate any continuous function.
    Reference: "Rational Neural Networks" (Boullé et al., 2020)
    """
    
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        # Initialize coefficients for P(x) = x + ...
        self.numerator = nn.Parameter(torch.randn(degree + 1) * 0.1)
        self.numerator.data[1] = 1.0  # Start as identity-ish
        self.denominator = nn.Parameter(torch.randn(degree + 1) * 0.1)
        self.denominator.data[0] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P(x) = sum a_i x^i
        # Q(x) = sum b_i x^i
        x_powers = torch.stack([x**i for i in range(self.degree + 1)], dim=-1)
        num = (x_powers * self.numerator).sum(dim=-1)
        den = (x_powers * self.denominator).sum(dim=-1)
        return num / (den + 1e-6)


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation with learnable parameters.
    
    u(x) = a * act(b * x + c) + d
    
    Can adapt slope, shift, and scale during training.
    """
    
    def __init__(
        self,
        base_activation: nn.Module = nn.Tanh(),
        init_a: float = 1.0,
        init_b: float = 1.0,
        init_c: float = 0.0,
        init_d: float = 0.0,
    ):
        super().__init__()
        self.base_activation = base_activation
        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.d = nn.Parameter(torch.tensor(init_d))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * self.base_activation(self.b * x + self.c) + self.d


class PeriodicActivation(nn.Module):
    """
    General periodic activation: sum of sines/cosines.
    
    Useful for problems with known periodicity.
    """
    
    def __init__(self, num_frequencies: int = 4, learnable_freq: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        if learnable_freq:
            self.frequencies = nn.Parameter(torch.arange(1, num_frequencies + 1).float())
        else:
            self.register_buffer("frequencies", torch.arange(1, num_frequencies + 1).float())
        
        self.coeffs_sin = nn.Parameter(torch.randn(num_frequencies) * 0.1)
        self.coeffs_cos = nn.Parameter(torch.randn(num_frequencies) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) or (N,)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Compute sum of sinusoids
        freq_x = x * self.frequencies  # (N, D, F) or (N, F)
        out = (self.coeffs_sin * torch.sin(freq_x) + self.coeffs_cos * torch.cos(freq_x)).sum(dim=-1)
        return out


class WaveletActivation(nn.Module):
    """
    Wavelet-based activation (Morlet wavelet).
    
    psi(x) = cos(5x) * exp(-x^2/2)
    """
    
    def __init__(self, frequency: float = 5.0, scale: float = 1.0):
        super().__init__()
        self.frequency = frequency
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.frequency * x) * torch.exp(-x**2 / (2 * self.scale**2))


# Activation function registry
ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sin": SinActivation,
    "swish": SwishActivation,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "gaussian": GaussianActivation,
    "rational": RationalActivation,
    "periodic": PeriodicActivation,
    "wavelet": WaveletActivation,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation module by name."""
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name](**kwargs)


def analyze_activation(activation: nn.Module, x_range: tuple = (-5, 5), num_points: int = 1000):
    """
    Analyze activation function properties.
    
    Returns:
        dict with keys: 'x', 'y', 'dy', 'd2y', 'range', 'saturation_points'
    """
    x = torch.linspace(x_range[0], x_range[1], num_points, requires_grad=True)
    y = activation(x)
    
    dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    d2y = torch.autograd.grad(dy.sum(), x, create_graph=True)[0]
    
    return {
        "x": x.detach(),
        "y": y.detach(),
        "dy": dy.detach(),
        "d2y": d2y.detach(),
        "range": (y.min().item(), y.max().item()),
        "saturation_points": x[(torch.abs(dy) < 1e-3)].detach(),
    }


__all__ = [
    "SinActivation",
    "SwishActivation",
    "GaussianActivation",
    "RationalActivation",
    "AdaptiveActivation",
    "PeriodicActivation",
    "WaveletActivation",
    "get_activation",
    "analyze_activation",
]