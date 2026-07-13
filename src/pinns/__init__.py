"""
PINNs Journey - Core Library

A modular, well-documented implementation of Physics-Informed Neural Networks
built from first principles using raw PyTorch.

Modules:
    config: Configuration management
    models: Neural network architectures
    losses: Physics, boundary, initial, and composite losses
    sampling: Collocation and boundary point sampling strategies
    equations: PDE/ODE definitions
    training: Training loops, optimizers, callbacks
    evaluation: Metrics and visualization
    utils: Seeding, logging, autograd utilities
"""

from .config import Config, load_config
from .utils.seeding import set_seed, get_seed
from .utils.logging import setup_logging, get_logger

__version__ = "0.1.0"
__author__ = "PINNs Journey"
__all__ = [
    "Config",
    "load_config",
    "set_seed",
    "get_seed",
    "setup_logging",
    "get_logger",
]