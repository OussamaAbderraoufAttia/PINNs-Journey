"""
Configuration Management

Provides structured configuration with validation using Hydra/OmegaConf.
Supports YAML configs, CLI overrides, and environment variables.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import omegaconf
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    activation: str = "tanh"  # tanh, sin, swish, relu, gelu
    output_activation: Optional[str] = None
    initialization: str = "xavier_normal"  # xavier_normal, xavier_uniform, he_normal, he_uniform
    dropout: float = 0.0
    batch_norm: bool = False
    input_dim: int = 1
    output_dim: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # adam, adamw, lbfgs
    scheduler: str = "step"  # step, cosine, plateau, none
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clip: float = 1.0
    early_stopping_patience: int = 1000
    early_stopping_min_delta: float = 1e-6
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "physics": 1.0,
        "boundary": 1.0,
        "initial": 1.0,
        "data": 1.0,
    })


@dataclass
class SamplingConfig:
    """Collocation and boundary point sampling configuration."""
    n_collocation: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000
    n_data: int = 0
    sampling_strategy: str = "uniform"  # uniform, latin_hypercube, sobol, adaptive
    domain: Dict[str, List[float]] = field(default_factory=dict)
    adaptive_sampling: bool = False
    adaptive_frequency: int = 1000
    adaptive_top_k: int = 1000


@dataclass
class EquationConfig:
    """PDE/ODE equation parameters."""
    equation_type: str = "heat"  # heat, burgers, wave, poisson, reaction_diffusion, ode
    parameters: Dict[str, float] = field(default_factory=dict)
    domain: Dict[str, List[float]] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    initial_condition: Dict[str, Any] = field(default_factory=dict)
    analytical_solution: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging and visualization configuration."""
    log_dir: str = "logs"
    log_interval: int = 100
    plot_interval: int = 500
    save_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "pinns-journey"
    wandb_entity: Optional[str] = None
    save_best_only: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "pinn_experiment"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    equation: EquationConfig = field(default_factory=EquationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class Config:
    """
    Configuration manager with validation and merging capabilities.
    
    Example:
        >>> config = Config.from_yaml("configs/heat_equation.yaml")
        >>> config.model.hidden_layers = [128, 128, 128]
        >>> config.save("configs/modified.yaml")
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self._config = config or ExperimentConfig()
    
    @property
    def model(self) -> ModelConfig:
        return self._config.model
    
    @property
    def training(self) -> TrainingConfig:
        return self._config.training
    
    @property
    def sampling(self) -> SamplingConfig:
        return self._config.sampling
    
    @property
    def equation(self) -> EquationConfig:
        return self._config.equation
    
    @property
    def logging(self) -> LoggingConfig:
        return self._config.logging
    
    @property
    def seed(self) -> int:
        return self._config.seed
    
    @property
    def device(self) -> str:
        return self._config.device
    
    @property
    def name(self) -> str:
        return self._config.name
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            raw_config = OmegaConf.load(f)
        
        # Convert to structured config with defaults
        structured = OmegaConf.structured(ExperimentConfig)
        merged = OmegaConf.merge(structured, raw_config)
        config = OmegaConf.to_object(merged)
        
        return cls(config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        structured = OmegaConf.structured(ExperimentConfig)
        merged = OmegaConf.merge(structured, config_dict)
        config = OmegaConf.to_object(merged)
        return cls(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(OmegaConf.structured(self._config), resolve=True)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=OmegaConf.structured(self._config), f=path)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        current = self.to_dict()
        self._deep_update(current, updates)
        self._config = OmegaConf.to_object(OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig), current
        ))
    
    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def __repr__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self._config))


def load_config(path: Union[str, Path]) -> Config:
    """Convenience function to load configuration."""
    return Config.from_yaml(path)


def create_default_configs(output_dir: Union[str, Path] = "configs") -> None:
    """Create default configuration files for each equation type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default ODE config
    ode_config = Config()
    ode_config._config.name = "ode_exponential_decay"
    ode_config._config.equation.equation_type = "ode"
    ode_config._config.equation.parameters = {"decay_rate": 1.0}
    ode_config._config.equation.domain = {"x": [0.0, 2.0]}
    ode_config._config.equation.initial_condition = {"x": 0.0, "y": 1.0}
    ode_config._config.equation.analytical_solution = "exp(-x)"
    ode_config.to_yaml(output_dir / "ode_exponential_decay.yaml")
    
    # Default Heat Equation config
    heat_config = Config()
    heat_config._config.name = "heat_equation_1d"
    heat_config._config.equation.equation_type = "heat"
    heat_config._config.equation.parameters = {"alpha": 0.01}
    heat_config._config.equation.domain = {"x": [0.0, 1.0], "t": [0.0, 1.0]}
    heat_config._config.equation.boundary_conditions = {
        "left": {"type": "dirichlet", "value": 0.0},
        "right": {"type": "dirichlet", "value": 0.0},
    }
    heat_config._config.equation.initial_condition = {
        "type": "sinusoidal",
        "expression": "sin(pi * x)"
    }
    heat_config._config.equation.analytical_solution = "exp(-alpha * pi^2 * t) * sin(pi * x)"
    heat_config.to_yaml(output_dir / "heat_equation_1d.yaml")
    
    # Default Burgers Equation config
    burgers_config = Config()
    burgers_config._config.name = "burgers_equation_1d"
    burgers_config._config.equation.equation_type = "burgers"
    burgers_config._config.equation.parameters = {"nu": 0.01 / 3.14159}
    burgers_config._config.equation.domain = {"x": [-1.0, 1.0], "t": [0.0, 1.0]}
    burgers_config._config.equation.boundary_conditions = {
        "left": {"type": "dirichlet", "value": 0.0},
        "right": {"type": "dirichlet", "value": 0.0},
    }
    burgers_config._config.equation.initial_condition = {
        "type": "sinusoidal",
        "expression": "-sin(pi * x)"
    }
    burgers_config._config.equation.analytical_solution = "exact_solution_available"
    burgers_config.to_yaml(output_dir / "burgers_equation_1d.yaml")
    
    print(f"Default configs created in {output_dir}/")


if __name__ == "__main__":
    create_default_configs()