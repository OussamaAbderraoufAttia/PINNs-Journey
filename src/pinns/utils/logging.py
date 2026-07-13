"""
Structured Logging Utilities

Provides consistent logging across the project with support for
console, file, and experiment tracking (Weights & Biases, TensorBoard).
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure project-wide logging.
    
    Args:
        log_dir: Directory for log files (None = no file logging)
        level: Logging level
        format_string: Custom format string
        use_colors: Whether to use colored console output
    
    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with optional colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors and sys.stdout.isatty():
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pinns_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)
    
    return root_logger


class ColoredFormatter(logging.Formatter):
    """Colored console output for different log levels."""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class TrainingLogger:
    """
    Structured training logger with multiple backends.
    
    Supports:
    - Console logging (with colors)
    - File logging
    - Weights & Biases
    - TensorBoard
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger(name)
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Initialize backends
        self._wandb = None
        self._tb_writer = None
        
        if use_wandb:
            self._init_wandb(wandb_config or {})
        
        if use_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(self, config: Dict[str, Any]) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=config.get("project", "pinns-journey"),
                entity=config.get("entity"),
                name=config.get("name"),
                config=config.get("config", {}),
                dir=str(self.log_dir) if self.log_dir else None,
            )
        except ImportError:
            self.logger.warning("wandb not installed, skipping W&B logging")
            self.use_wandb = False
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.log_dir / "tensorboard" if self.log_dir else "runs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir)
        except ImportError:
            self.logger.warning("tensorboard not installed, skipping TensorBoard logging")
            self.use_tensorboard = False
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all backends."""
        # Console/file logging
        metric_str = " | ".join(f"{k}: {v:.6e}" for k, v in metrics.items())
        self.logger.info(f"Step {step} | {prefix}{metric_str}")
        
        # WandB
        if self.use_wandb and self._wandb:
            prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
            self._wandb.log(prefixed, step=step)
        
        # TensorBoard
        if self.use_tensorboard and self._tb_writer:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(f"{prefix}{k}", v, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.use_wandb and self._wandb:
            self._wandb.config.update(params)
        if self.use_tensorboard and self._tb_writer:
            self._tb_writer.add_hparams(params, {})
    
    def log_figure(self, name: str, figure, step: int) -> None:
        """Log matplotlib figure."""
        if self.use_wandb and self._wandb:
            import wandb
            self._wandb.log({name: wandb.Image(figure)}, step=step)
        if self.use_tensorboard and self._tb_writer:
            self._tb_writer.add_figure(name, figure, step)
    
    def log_model(self, model: "torch.nn.Module", step: int) -> None:
        """Log model graph and parameters."""
        if self.use_tensorboard and self._tb_writer:
            # Try to log model graph
            try:
                dummy_input = torch.randn(1, model.input_dim)
                self._tb_writer.add_graph(model, dummy_input)
            except Exception:
                pass
    
    def close(self) -> None:
        """Close all logging backends."""
        if self.use_wandb and self._wandb:
            self._wandb.finish()
        if self.use_tensorboard and self._tb_writer:
            self._tb_writer.close()


import torch  # Import here to avoid circular import


__all__ = [
    "setup_logging",
    "get_logger",
    "TrainingLogger",
]