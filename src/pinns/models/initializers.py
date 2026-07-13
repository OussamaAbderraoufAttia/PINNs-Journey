"""
Weight Initialization Schemes

Various initialization strategies for PINNs, including specialized
initializations for physics-informed learning.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Optional


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier normal initialization."""
    return init.xavier_normal_(tensor, gain=gain)


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier uniform initialization."""
    return init.xavier_uniform_(tensor, gain=gain)


def he_normal_(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> torch.Tensor:
    """He (Kaiming) normal initialization."""
    return init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def he_uniform_(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> torch.Tensor:
    """He (Kaiming) uniform initialization."""
    return init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Orthogonal initialization."""
    return init.orthogonal_(tensor, gain=gain)


def siren_uniform_(tensor: torch.Tensor, w0: float = 30.0, is_first: bool = False) -> torch.Tensor:
    """
    SIREN initialization for sine activations.
    
    First layer: U(-1/n, 1/n)
    Other layers: U(-sqrt(6/n)/w0, sqrt(6/n)/w0)
    """
    fan_in = tensor.shape[1]
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / w0
    return init.uniform_(tensor, -bound, bound)


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    """LeCun normal initialization (for SELU)."""
    fan_in = tensor.shape[1]
    std = math.sqrt(1.0 / fan_in)
    return init.normal_(tensor, mean=0.0, std=std)


def delta_orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Delta-orthogonal initialization for convolutional layers.
    Places orthogonal matrix at center of filter.
    """
    if tensor.ndimension() < 3:
        raise ValueError("Tensor must have at least 3 dimensions")
    rows, cols = tensor.shape[0], tensor.shape[1]
    if rows != cols:
        raise ValueError("Delta orthogonal only works for square matrices")
    
    # Create orthogonal matrix
    q = torch.empty(rows, cols).normal_(0, 1)
    q, _ = torch.qr(q)
    q *= gain
    
    # Place at center
    with torch.no_grad():
        tensor.zero_()
        mid = [s // 2 for s in tensor.shape[2:]]
        for i in range(rows):
            for j in range(cols):
                if i == j:
                    idx = [i, j] + mid
                    tensor[idx] = q[i, j]
    return tensor


def physics_informed_init_(
    tensor: torch.Tensor,
    pde_order: int = 2,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Physics-informed initialization based on PDE order.
    
    For PDEs of order k, initialize weights to respect derivative scaling.
    Based on: "On the Initialization of Physics-Informed Neural Networks"
    """
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # Scale based on PDE order - higher order PDEs need smaller weights
    # to prevent gradient explosion in higher-order derivatives
    pde_scale = scale / (pde_order ** 0.5)
    
    # Modified Xavier for physics
    bound = math.sqrt(6.0 / (fan_in + fan_out)) * pde_scale
    return init.uniform_(tensor, -bound, bound)


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    """Zero initialization."""
    return init.zeros_(tensor)


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    """Ones initialization."""
    return init.ones_(tensor)


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    """Constant initialization."""
    return init.constant_(tensor, val)


class Initializer:
    """
    Unified initializer class for consistent weight initialization.
    
    Example:
        >>> initializer = Initializer("xavier_normal", gain=1.0)
        >>> initializer.apply(model)
    """
    
    SCHEMES = {
        "xavier_normal": xavier_normal_,
        "xavier_uniform": xavier_uniform_,
        "he_normal": he_normal_,
        "he_uniform": he_uniform_,
        "orthogonal": orthogonal_,
        "siren": siren_uniform_,
        "lecun_normal": lecun_normal_,
        "physics_informed": physics_informed_init_,
        "zeros": zeros_,
        "ones": ones_,
    }
    
    def __init__(
        self,
        scheme: str = "xavier_normal",
        gain: float = 1.0,
        w0: float = 30.0,
        pde_order: int = 2,
        scale: float = 1.0,
    ):
        if scheme not in self.SCHEMES:
            raise ValueError(f"Unknown scheme: {scheme}. Options: {list(self.SCHEMES.keys())}")
        
        self.scheme = scheme
        self.gain = gain
        self.w0 = w0
        self.pde_order = pde_order
        self.scale = scale
    
    def __call__(self, tensor: torch.Tensor, is_first: bool = False) -> torch.Tensor:
        """Apply initialization to tensor."""
        fn = self.SCHEMES[self.scheme]
        
        if self.scheme == "siren":
            return fn(tensor, w0=self.w0, is_first=is_first)
        elif self.scheme == "physics_informed":
            return fn(tensor, pde_order=self.pde_order, scale=self.scale)
        elif self.scheme in ["xavier_normal", "xavier_uniform", "orthogonal"]:
            return fn(tensor, gain=self.gain)
        elif self.scheme in ["he_normal", "he_uniform"]:
            return fn(tensor, nonlinearity="tanh")
        else:
            return fn(tensor)
    
    def apply(self, module: nn.Module) -> None:
        """Apply initialization to all linear layers in module."""
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                is_first = "0.weight" in name or "input_layer.weight" in name
                self(param, is_first=is_first)
            elif "bias" in name:
                init.zeros_(param)


def calculate_gain(activation: str) -> float:
    """Calculate gain for initialization based on activation function."""
    activation = activation.lower()
    if activation in ["tanh", "sigmoid"]:
        return 5.0 / 3.0  # ~1.67
    elif activation in ["relu", "leaky_relu"]:
        return math.sqrt(2.0)
    elif activation in ["selu"]:
        return 0.75
    elif activation in ["sin", "siren"]:
        return 1.0
    elif activation in ["swish", "silu"]:
        return 1.0
    elif activation in ["gelu"]:
        return 1.0
    else:
        return 1.0


def initialize_model(
    model: nn.Module,
    scheme: str = "xavier_normal",
    activation: str = "tanh",
    pde_order: int = 2,
) -> None:
    """
    Convenience function to initialize entire model.
    
    Args:
        model: PyTorch module
        scheme: Initialization scheme
        activation: Activation function (for gain calculation)
        pde_order: PDE order (for physics_informed scheme)
    """
    gain = calculate_gain(activation)
    initializer = Initializer(
        scheme=scheme,
        gain=gain,
        pde_order=pde_order,
    )
    initializer.apply(model)


__all__ = [
    "xavier_normal_",
    "xavier_uniform_",
    "he_normal_",
    "he_uniform_",
    "orthogonal_",
    "siren_uniform_",
    "lecun_normal_",
    "delta_orthogonal_",
    "physics_informed_init_",
    "zeros_",
    "ones_",
    "constant_",
    "Initializer",
    "calculate_gain",
    "initialize_model",
]