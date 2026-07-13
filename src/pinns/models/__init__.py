"""
Models Package

Neural network architectures for PINNs.
"""

from .mlp import (
    MLP,
    ResidualMLP,
    FourierFeatureMLP,
    SIREN,
    SineLayer,
    ModifiedMLP,
    ActivationType,
    get_activation,
    count_parameters,
    model_summary,
)

from .initializers import (
    Initializer,
    initialize_model,
    xavier_normal_,
    xavier_uniform_,
    he_normal_,
    he_uniform_,
    orthogonal_,
    siren_uniform_,
    physics_informed_init_,
    calculate_gain,
)

from .activations import (
    SinActivation,
    SwishActivation,
    GaussianActivation,
    RationalActivation,
    AdaptiveActivation,
    PeriodicActivation,
    WaveletActivation,
    get_activation as get_activation_fn,
    analyze_activation,
)

__all__ = [
    # MLP
    "MLP",
    "ResidualMLP",
    "FourierFeatureMLP",
    "SIREN",
    "SineLayer",
    "ModifiedMLP",
    "ActivationType",
    "get_activation",
    "count_parameters",
    "model_summary",
    # Initializers
    "Initializer",
    "initialize_model",
    "xavier_normal_",
    "xavier_uniform_",
    "he_normal_",
    "he_uniform_",
    "orthogonal_",
    "siren_uniform_",
    "physics_informed_init_",
    "calculate_gain",
    # Activations
    "SinActivation",
    "SwishActivation",
    "GaussianActivation",
    "RationalActivation",
    "AdaptiveActivation",
    "PeriodicActivation",
    "WaveletActivation",
    "get_activation_fn",
    "analyze_activation",
]