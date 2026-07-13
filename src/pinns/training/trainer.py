"""
Main Training Module

Trainer class for PINNs with callbacks, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10000
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    early_stopping_patience: int = 1000
    log_freq: int = 100
    save_freq: int = 1000
    eval_freq: int = 500
    n_collocation: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass
    
    def on_batch_end(self, trainer: "Trainer", epoch: int, batch_idx: int, losses: Dict) -> None:
        pass
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class LoggingCallback(Callback):
    """Log training progress."""
    
    def __init__(self, log_freq: int = 100, logger: logging.Logger = None):
        self.log_freq = log_freq
        self.logger = logger or logging.getLogger(__name__)
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        if epoch % self.log_freq == 0:
            loss_str = " | ".join(f"{k}: {v:.6e}" for k, v in losses.items())
            self.logger.info(f"Epoch {epoch:5d} | {loss_str}")


class CheckpointCallback(Callback):
    """Save model checkpoints."""
    
    def __init__(self, save_dir: str, save_freq: int = 1000, save_best: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_loss = float('inf')
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        total_loss = losses.get('total', float('inf'))
        
        # Save periodic checkpoint
        if epoch % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(path)
        
        # Save best model
        if self.save_best and total_loss < self.best_loss:
            self.best_loss = total_loss
            path = self.save_dir / "best_model.pt"
            trainer.save_checkpoint(path)


class EarlyStoppingCallback(Callback):
    """Stop training when loss stops improving."""
    
    def __init__(self, patience: int = 1000, min_delta: float = 1e-6, monitor: str = "total"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_loss = float('inf')
        self.counter = 0
        self.stopped = False
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, losses: Dict) -> None:
        current = losses.get(self.monitor, float('inf'))
        
        if current < self.best_loss - self.min_delta:
            self.best_loss = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.logger.info(f"Early stopping at epoch {epoch}")
                self.stopped = True


class Trainer:
    """
    PINN Trainer with callbacks, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        equation,
        loss_fn,
        config: TrainingConfig,
        callbacks: List[Callback] = None,
        device: str = "auto",
    ):
        self.model = model
        self.equation = equation
        self.loss_fn = loss_fn
        self.config = config
        self.callbacks = callbacks or []
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # State
        self.current_epoch = 0
        self.history = {k: [] for k in ['physics', 'boundary', 'initial', 'data', 'total']}
        self.logger = logging.getLogger(__name__)
        
        # Default callbacks
        self.callbacks.extend([
            LoggingCallback(config.log_freq, self.logger),
        ])
    
    def _create_optimizer(self) -> optim.Optimizer:
        name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        
        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr)
        elif name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif name == "lbfgs":
            return optim.LBFGS(
                self.model.parameters(),
                lr=lr,
                max_iter=20,
                line_search_fn="strong_wolfe"
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _create_scheduler(self):
        name = self.config.scheduler.lower()
        
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif name == "cosine_warm_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.epochs//10)
        elif name == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.epochs//3, gamma=0.1)
        elif name == "exponential":
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        elif name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=500, factor=0.5)
        elif name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.learning_rate * 10,
                total_steps=self.config.epochs
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        self._call_callbacks("on_train_begin")
        
        self.logger.info(f"Training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            self._call_callbacks("on_epoch_begin", epoch)
            
            # Training step
            losses = self._train_epoch()
            
            # Record history
            for k, v in losses.items():
                if k in self.history:
                    self.history[k].append(v)
            
            self._call_callbacks("on_epoch_end", epoch, losses)
            
            # Check early stopping
            if any(cb.stopped for cb in self.callbacks if isinstance(cb, EarlyStoppingCallback)):
                break
        
        self._call_callbacks("on_train_end")
        
        return self.history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample points
        coords = self._sample_points()
        coords = {k: v.to(self.device) for k, v in coords.items()}
        
        # Compute loss
        losses = self.loss_fn(self.model, coords)
        
        total_loss = losses.get('total', sum(losses.values()))
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        if isinstance(self.optimizer, optim.LBFGS):
            def closure():
                self.optimizer.zero_grad()
                losses = self.loss_fn(self.model, coords)
                total_loss = losses.get('total', sum(losses.values()))
                total_loss.backward()
                return total_loss
            self.optimizer.step(closure)
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        # Return detached losses
        return {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
    
    def _sample_points(self) -> Dict[str, Tensor]:
        """Sample training points."""
        coords = {}
        
        # Collocation points
        phys_coords = self.equation.sample_domain(self.config.n_collocation if hasattr(self.config, 'n_collocation') else 10000)
        for k, v in phys_coords.items():
            coords[k] = v
        
        # Boundary points
        if self.equation.boundary_conditions:
            bc_coords = self.equation.sample_boundary(
                getattr(self.config, 'n_boundary', 1000)
            )
            for k, v in bc_coords.items():
                if k in coords:
                    coords[k] = torch.cat([coords[k], v], dim=0)
                else:
                    coords[k] = v
        
        # Initial points
        if self.equation.initial_conditions:
            ic_coords = self.equation.sample_initial(
                getattr(self.config, 'n_initial', 1000)
            )
            for k, v in ic_coords.items():
                if k in coords:
                    coords[k] = torch.cat([coords[k], v], dim=0)
                else:
                    coords[k] = v
        
        return coords
    
    def _call_callbacks(self, method: str, *args, **kwargs):
        """Call method on all callbacks."""
        for cb in self.callbacks:
            getattr(cb, method)(self, *args, **kwargs)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config.__dict__,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
    
    def evaluate(self, test_coords: Dict[str, Tensor]) -> Dict[str, float]:
        """Evaluate model on test points."""
        from src.pinns.evaluation import evaluate_model
        return evaluate_model(self.model, self.equation, test_coords)


def create_trainer(
    model: nn.Module,
    equation,
    loss_fn,
    config: TrainingConfig = None,
    **kwargs
) -> Trainer:
    """Factory function to create trainer."""
    config = config or TrainingConfig()
    return Trainer(model, equation, loss_fn, config, **kwargs)