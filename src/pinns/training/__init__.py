"""
Training Package

Training loop, optimizers, schedulers, and callbacks for PINNs.
"""

from .trainer import (
    TrainingConfig,
    Callback,
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    Trainer,
)

__all__ = [
    "TrainingConfig",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "Trainer",
]
