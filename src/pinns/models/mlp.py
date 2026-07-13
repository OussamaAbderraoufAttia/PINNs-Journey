"""
Neural Network Models

Core neural network architectures for PINNs, including MLPs,
activation functions, and weight initialization schemes.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional, Callable, Union
from enum import Enum


class ActivationType(Enum):
    """Supported activation functions."""
    TANH = "tanh"
    SIN = "sin"
    SWISH = "swish"
    GELU = "gelu"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SOFTPLUS = "softplus"
    SIGMOID = "sigmoid"


def get_activation(activation: Union[str, ActivationType]) -> nn.Module:
    """
    Get activation function module by name.
    
    Args:
        activation: Activation function name or enum
    
    Returns:
        Activation module
    """
    if isinstance(activation, ActivationType):
        activation = activation.value
    
    activation = activation.lower()
    
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "sin":
        return SinActivation()
    elif activation == "swish":
        return nn.SiLU()  # Swish = SiLU
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.01)
    elif activation == "elu":
        return nn.ELU()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class SinActivation(nn.Module):
    """
    Sinusoidal activation function: sin(x)
    
    Particularly effective for PINNs due to spectral bias properties
    and ability to represent high-frequency functions.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SwishActivation(nn.Module):
    """
    Swish activation: x * sigmoid(x)
    
    Self-gated activation that can outperform ReLU in deep networks.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for PINNs.
    
    Features:
    - Configurable depth, width, activation
    - Multiple weight initialization schemes
    - Optional layer normalization
    - Optional residual connections
    - Input/output normalization support
    
    Example:
        >>> model = MLP(
        ...     input_dim=2,
        ...     output_dim=1,
        ...     hidden_dims=[64, 64, 64],
        ...     activation="tanh",
        ...     init_type="xavier_normal"
        ... )
        >>> x = torch.randn(100, 2)
        >>> y = model(x)  # Shape: (100, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Union[str, ActivationType] = "tanh",
        output_activation: Optional[Union[str, ActivationType]] = None,
        init_type: str = "xavier_normal",
        use_layer_norm: bool = False,
        use_residual: bool = False,
        dropout: float = 0.0,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation if isinstance(activation, str) else activation.value
        self.use_residual = use_residual
        self.input_scale = input_scale
        self.output_scale = output_scale
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)
            
            # Layer norm
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            act = get_activation(activation)
            layers.append(act)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        output_linear = nn.Linear(prev_dim, output_dim)
        layers.append(output_linear)
        
        # Output activation
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda m: self._init_weights(m, init_type))
        
        # Store input dimension for graph logging
        self.register_buffer("input_dim_buffer", torch.tensor(input_dim))
    
    @property
    def input_dim(self) -> int:
        return self.input_dim_buffer.item()
    
    def _init_weights(self, module: nn.Module, init_type: str) -> None:
        """Initialize weights according to scheme."""
        if isinstance(module, nn.Linear):
            if init_type == "xavier_normal":
                init.xavier_normal_(module.weight)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(module.weight)
            elif init_type == "he_normal":
                init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif init_type == "he_uniform":
                init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight)
            elif init_type == "normal":
                init.normal_(module.weight, mean=0.0, std=0.1)
            elif init_type == "uniform":
                init.uniform_(module.weight, a=-0.1, b=0.1)
            elif init_type == "siren":
                # SIREN initialization for sinusoidal activations
                init.uniform_(module.weight, a=-1.0, b=1.0)
                with torch.no_grad():
                    module.weight *= 30.0 / module.weight.shape[1]
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional input/output scaling."""
        x = x * self.input_scale
        out = self.network(x)
        return out * self.output_scale


class ResidualMLP(nn.Module):
    """
    Residual MLP with skip connections.
    
    Better gradient flow for very deep networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: Union[str, ActivationType] = "tanh",
        init_type: str = "xavier_normal",
    ):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.apply(lambda m: self._init_weights(m, init_type))
    
    def _init_weights(self, module: nn.Module, init_type: str) -> None:
        if isinstance(module, nn.Linear):
            if init_type == "xavier_normal":
                init.xavier_normal_(module.weight)
            elif init_type == "he_normal":
                init.kaiming_normal_(module.weight, nonlinearity="tanh")
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x
            x = self.activation(layer(x)) + residual
        return self.output_layer(x)


class FourierFeatureMLP(nn.Module):
    """
    MLP with Fourier feature mapping for high-frequency functions.
    
    Maps inputs to high-dimensional Fourier features before MLP:
    γ(x) = [cos(2πBx), sin(2πBx)] where B ~ N(0, σ²)
    
    Reference: "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains" (Tancik et al., 2020)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        mapping_size: int = 256,
        sigma: float = 10.0,
        activation: Union[str, ActivationType] = "relu",
        init_type: str = "xavier_normal",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma
        
        # Random Fourier feature mapping (fixed, not learned)
        self.register_buffer(
            "B",
            torch.randn(mapping_size, input_dim) * sigma
        )
        
        # MLP takes 2 * mapping_size inputs (sin and cos)
        mlp_input_dim = 2 * mapping_size
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier feature mapping
        # x: (N, D) -> (N, mapping_size) @ (mapping_size, D)^T = (N, mapping_size)
        x_proj = 2 * torch.pi * x @ self.B.T  # (N, mapping_size)
        x_fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # (N, 2*mapping_size)
        return self.mlp(x_fourier)


class SIREN(nn.Module):
    """
    SIREN: Implicit Neural Representations with Periodic Activation Functions.
    
    Uses sine activation with special initialization for representing
    complex signals and derivatives.
    
    Reference: "Implicit Neural Representations with Periodic Activation Functions"
    (Sitzmann et al., 2020)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        w0: float = 30.0,
        w0_initial: float = 30.0,
    ):
        super().__init__()
        
        self.w0 = w0
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            is_first = (i == 0)
            layer = SineLayer(
                prev_dim, hidden_dim,
                is_first=is_first,
                w0=w0_initial if is_first else w0
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer (linear, no sine)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self._init_output_layer()
    
    def _init_output_layer(self):
        with torch.no_grad():
            bound = torch.sqrt(torch.tensor(6.0 / self.output_layer.in_features)) / self.w0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class SineLayer(nn.Module):
    """Single sine layer with SIREN initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool,
        w0: float = 30.0,
    ):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = torch.sqrt(torch.tensor(6.0 / self.linear.in_features)) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class ModifiedMLP(nn.Module):
    """
    Modified MLP with feature embedding and gating (like MIONet/DeepONet trunk).
    
    Architecture: u = σ(W_z z + b_z) ⊙ σ(W_x x + b_x)
    where z = σ(W_e x + b_e) is an embedding.
    
    Helps with gradient propagation in deep PINNs.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        embedding_dim: int = 64,
        activation: Union[str, ActivationType] = "tanh",
        init_type: str = "xavier_normal",
    ):
        super().__init__()
        
        self.embedding = MLP(
            input_dim=input_dim,
            output_dim=embedding_dim,
            hidden_dims=[embedding_dim],
            activation=activation,
            init_type=init_type,
        )
        
        self.trunk = MLP(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            init_type=init_type,
        )
        
        self.gate = nn.Linear(embedding_dim, hidden_dims[-1])
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
        self.apply(lambda m: self._init_weights(m, init_type))
    
    def _init_weights(self, module: nn.Module, init_type: str):
        if isinstance(module, nn.Linear):
            if init_type == "xavier_normal":
                init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        u = self.trunk(x)
        gate = torch.sigmoid(self.gate(z))
        return self.output(u * gate)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 2)) -> str:
    """Generate model summary string."""
    from torchinfo import summary
    try:
        return str(summary(model, input_size=input_size, verbose=0))
    except ImportError:
        # Fallback summary
        total_params = count_parameters(model)
        layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
        return f"Model: {model.__class__.__name__}\nLayers: {layers}\nParameters: {total_params:,}"


__all__ = [
    "ActivationType",
    "get_activation",
    "SinActivation",
    "SwishActivation",
    "MLP",
    "ResidualMLP",
    "FourierFeatureMLP",
    "SIREN",
    "SineLayer",
    "ModifiedMLP",
    "count_parameters",
    "model_summary",
]