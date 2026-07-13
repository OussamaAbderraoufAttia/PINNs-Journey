"""
Training Package

Training loop, optimizers, schedulers, and callbacks for PINNs.
"""

from .trainer import (
    TrainingConfig,
    OptimizerFactory,
    SchedulerFactory,
    Callback,
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LRLoggerCallback,
    Trainer,
    create_trainer,
)

__all__ = [
    "TrainingConfig",
    "OptimizerFactory",
    "SchedulerFactory",
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LRLoggerCallback",
    "Trainer",
    "create_trainer",
]