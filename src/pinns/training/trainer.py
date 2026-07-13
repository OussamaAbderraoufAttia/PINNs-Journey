"""
Training Module for PINNs

Provides training loop, optimizers, schedulers, and callbacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10000
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "cosine"
    scheduler_params: Dict = None
    gradient_clip: float = 1.0
    log_freq: int = 100
    eval_freq: int = 500
    save_freq: int = 1000
    early_stopping_patience: int = 1000
    early_stopping_min_delta: float = 1e-6
    device: str = "auto"
    seed: int = 42


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(
        name: str,
        model: nn.Module,
        lr: float,
        **kwargs
    ) -> optim.Optimizer:
        name = name.lower()
        
        if name == "adam":
            return optim.Adam(model.parameters(), lr=lr, **kwargs)
        elif name == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, **kwargs)
        elif name == "sgd":
            return optim.SGD(model.parameters(), lr=lr, **kwargs)
        elif name == "lbfgs":
            return optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=20,
                line_search_fn="strong_wolfe",
                **kwargs
            )
        elif name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {name}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create(
        name: str,
        optimizer: optim.Optimizer,
        epochs: int,
        **kwargs
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        name = name.lower()
        
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **kwargs)
        elif name == "cosine_warm_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs//10, **kwargs)
        elif name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1, **kwargs)
        elif name == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, **kwargs)
        elif name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, **kwargs)
        elif name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=kwargs.get("max_lr", 1e-3),
                total_steps=epochs, **kwargs
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass
    
    @abstractmethod
    def on_batch_end(self, trainer: "Trainer", epoch: int, batch_idx: int, losses: Dict) -> None:
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class LoggingCallback(Callback):
    """Logs training progress."""
    
    def __init__(self, log_freq: int = 100, logger: logging.Logger = None):
        self.log_freq = log_freq
        self.logger = logger or logging.getLogger(__name__)
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        self.logger.info(f"Training started on {trainer.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass
    
    def on_batch_end(self, trainer: "Trainer", epoch: int, batch_idx: int, losses: Dict) -> None:
        if batch_idx % self.log_freq == 0:
            loss_str = " | ".join(f"{k}: {v:.6e}" for k, v in losses.items())
            self.logger.info(f"Epoch {epoch} | Batch {batch_idx} | {loss_str}")
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        if epoch % self.log_freq == 0:
            loss_str = " | ".join(f"{k}: {v:.6e}" for k, v in losses.items())
            self.logger.info(f"Epoch {epoch} | {loss_str}")
    
    def on_train_end(self, trainer: "Trainer") -> None:
        self.logger.info("Training completed")


class CheckpointCallback(Callback):
    """Saves model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        save_freq: int = 1000,
        save_best: bool = True,
        monitor: str = "total",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best = save_best
        self.monitor = monitor
        self.best_loss = float('inf')
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass
    
    def on_batch_end(self, trainer: "Trainer", epoch: int, batch_idx: int, losses: Dict) -> None:
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        # Save periodic checkpoint
        if epoch % self.save_freq == 0:
            self._save_checkpoint(trainer, epoch, losses, f"checkpoint_epoch_{epoch}.pt")
        
        # Save best model
        if self.save_best:
            current_loss = losses.get(self.monitor, losses.get("total", float('inf')))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._save_checkpoint(trainer, epoch, losses, "best_model.pt")
    
    def on_train_end(self, trainer: "Trainer") -> None:
        self._save_checkpoint(trainer, trainer.current_epoch, trainer.current_losses, "final_model.pt")
    
    def _save_checkpoint(
        self,
        trainer: "Trainer",
        epoch: int,
        losses: Dict,
        filename: str
    ):
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "losses": losses,
            "config": trainer.config.__dict__,
        }, path)
        trainer.logger.info(f"Checkpoint saved: {path}")


class EarlyStoppingCallback(Callback):
    """Stops training when monitored metric stops improving."""
    
    def __init__(
        self,
        patience: int = 1000,
        min_delta: float = 1e-6,
        monitor: str = "total",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else -float('inf')
        self.counter = 0
        self.stopped = False
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass
    
    def on_batch_end(self, trainer: "Trainer", epoch: int, batch_idx: int, losses: Dict) -> None:
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        current = losses.get(self.monitor, losses.get("total", float('inf')))
        
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.logger.info(f"Early stopping triggered at epoch {epoch}")
                self.stopped = True
    
    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class LRLoggerCallback(Callback):
    """Logs learning rate."""
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        if trainer.scheduler:
            lr = trainer.scheduler.get_last_lr()[0]
            trainer.logger.debug(f"Learning rate: {lr:.2e}")


class Trainer:
    """
    Main training class for PINNs.
    
    Handles:
    - Training loop with batching
    - Multiple loss components
    - Callbacks (logging, checkpointing, early stopping)
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        equation: Callable,
        loss_fn: Callable,
        config: TrainingConfig,
        callbacks: List[Callback] = None,
        train_data: Dict = None,
    ):
        self.model = model
        self.equation = equation
        self.loss_fn = loss_fn
        self.config = config
        self.callbacks = callbacks or []
        self.train_data = train_data
        
        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = OptimizerFactory.create(
            config.optimizer,
            model,
            config.learning_rate,
        )
        
        # Scheduler
        self.scheduler = SchedulerFactory.create(
            config.scheduler,
            self.optimizer,
            config.epochs,
            **(config.scheduler_params or {}),
        )
        
        # State
        self.current_epoch = 0
        self.current_losses = {}
        self.history = {"epoch": [], "losses": {}}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Add default callbacks if none provided
        if not self.callbacks:
            self.callbacks = [
                LoggingCallback(config.log_freq),
                CheckpointCallback("checkpoints", config.save_freq),
            ]
            if config.early_stopping_patience > 0:
                self.callbacks.append(
                    EarlyStoppingCallback(
                        config.early_stopping_patience,
                        config.early_stopping_min_delta,
                    )
                )
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        self._call_callbacks("on_train_begin")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            self._call_callbacks("on_epoch_begin", epoch)
            
            # Training epoch
            epoch_losses = self._train_epoch(epoch)
            self.current_losses = epoch_losses
            
            # Record history
            self.history["epoch"].append(epoch)
            for k, v in epoch_losses.items():
                if k not in self.history["losses"]:
                    self.history["losses"][k] = []
                self.history["losses"][k].append(v)
            
            self._call_callbacks("on_epoch_end", epoch, epoch_losses)
            
            # Check early stopping
            if any(cb.stopped for cb in self.callbacks if isinstance(cb, EarlyStoppingCallback)):
                break
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_losses.get("total", epoch_losses.get("physics", 0)))
                else:
                    self.scheduler.step()
        
        self._call_callbacks("on_train_end")
        
        return self.history
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        
        # Sample points
        coords = self._sample_training_points()
        
        # Move to device
        coords = {k: v.to(self.device) for k, v in coords.items()}
        
        # Forward + backward
        self.optimizer.zero_grad()
        
        losses = self.loss_fn(self.model, coords)
        
        total_loss = losses.get("total", sum(losses.values()))
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        
        # Call batch callbacks
        self._call_callbacks("on_batch_end", epoch, 0, losses)
        
        # Return detached losses
        return {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
    
    def _sample_training_points(self) -> Dict[str, Tensor]:
        """Sample training points for current epoch."""
        # Get domain bounds from equation
        bounds = self.equation.domain.bounds
        
        # Physics (collocation) points
        physics_coords = self.equation.sample_domain(self.config.n_collocation if hasattr(self.config, 'n_collocation') else 10000)
        
        # Boundary points
        bc_coords = {}
        if self.equation.boundary_conditions:
            bc_coords = self.equation.sample_boundary(
                getattr(self.config, 'n_boundary', 1000)
            )
        
        # Initial points
        ic_coords = {}
        if self.equation.initial_conditions:
            ic_coords = self.equation.sample_initial(
                getattr(self.config, 'n_initial', 1000)
            )
        
        # Combine all coordinates
        all_coords = {}
        for key in bounds.keys():
            tensors = []
            if key in physics_coords:
                tensors.append(physics_coords[key])
            if key in bc_coords:
                tensors.append(bc_coords[key])
            if key in ic_coords:
                tensors.append(ic_coords[key])
            
            if tensors:
                all_coords[key] = torch.cat(tensors, dim=0)
            else:
                # Fallback
                all_coords[key] = physics_coords.get(key, torch.empty(0, 1))
        
        return all_coords
    
    def _call_callbacks(self, method: str, *args, **kwargs):
        """Call method on all callbacks."""
        for callback in self.callbacks:
            getattr(callback, method)(self, *args, **kwargs)
    
    def evaluate(self, test_coords: Dict[str, Tensor]) -> Dict[str, float]:
        """Evaluate model on test points."""
        self.model.eval()
        test_coords = {k: v.to(self.device) for k, v in test_coords.items()}
        
        with torch.no_grad():
            losses = self.loss_fn(self.model, test_coords)
        
        return {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
    
    def predict(self, coords: Dict[str, Tensor]) -> Tensor:
        """Get model predictions."""
        self.model.eval()
        coords = {k: v.to(self.device) for k, v in coords.items()}
        x = self.equation.domain.to_tensor(coords)
        with torch.no_grad():
            return self.model(x).cpu()


def create_trainer(
    model: nn.Module,
    equation: Callable,
    loss_fn: Callable,
    config: TrainingConfig = None,
    **kwargs
) -> Trainer:
    """Factory function to create trainer."""
    config = config or TrainingConfig()
    return Trainer(model, equation, loss_fn, config, **kwargs)


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