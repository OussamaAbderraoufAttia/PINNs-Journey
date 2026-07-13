"""
Reproducibility Utilities

Ensures consistent results across runs by setting seeds for all random sources.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic algorithms (may impact performance)
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed() -> int:
    """Get the current PyTorch seed."""
    return torch.initial_seed()


def get_random_state() -> dict:
    """Get current random state from all sources."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_random_state(state: dict) -> None:
    """Restore random state from dictionary."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class SeedContext:
    """
    Context manager for temporary seed setting.
    
    Example:
        >>> with SeedContext(42):
        ...     x = torch.randn(10)
        >>> # Original seed restored
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.previous_state = None
    
    def __enter__(self):
        self.previous_state = get_random_state()
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_state:
            set_random_state(self.previous_state)


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader reproducibility.
    
    Usage:
        >>> loader = DataLoader(dataset, worker_init_fn=worker_init_fn)
    """
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "set_seed",
    "get_seed",
    "get_random_state",
    "set_random_state",
    "SeedContext",
    "worker_init_fn",
]