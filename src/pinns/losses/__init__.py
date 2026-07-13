"""
Losses Package

Physics-informed loss functions for PINNs.
"""

from .physics import (
    LossType,
    LossWeights,
    LossComponent,
    PhysicsLoss,
    BoundaryLoss,
    InitialLoss,
    DataLoss,
    RegularizationLoss,
    AdaptiveLossWeighting,
    CompositeLoss,
    create_pinn_loss,
)

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