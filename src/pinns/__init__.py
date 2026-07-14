"""
PINNs Journey - Main Package
"""

from .config import Config, ModelConfig, TrainingConfig as TrainingConfigDeprecated, SamplingConfig, EquationConfig, LossWeights, LoggingConfig
from .models import MLP, count_parameters, model_summary
from .equations import get_equation, list_equations, register_equation
from .losses import create_pinn_loss, PhysicsLoss, BoundaryLoss, InitialLoss, DataLoss
from .utils.derivatives import gradient, jacobian, hessian, laplacian, divergence
from .utils.seeding import set_seed, get_seed
from .utils.logging import setup_logging, get_logger, TrainingLogger
from .training import Trainer, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    "TrainingConfig",
    # Models
    "MLP",
    "count_parameters",
    "model_summary",
    # Equations
    "get_equation",
    "list_equations",
    "register_equation",
    # Losses
    "create_pinn_loss",
    "PhysicsLoss",
    "BoundaryLoss",
    "InitialLoss",
    "DataLoss",
    # Derivatives
    "gradient",
    "jacobian",
    "hessian",
    "laplacian",
    "divergence",
    # Training
    "Trainer",
    # Utilities
    "set_seed",
    "get_seed",
    "setup_logging",
    "get_logger",
    "TrainingLogger",
]
