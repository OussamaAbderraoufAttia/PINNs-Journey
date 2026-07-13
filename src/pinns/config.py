"""
Configuration Management

YAML-based configuration with dataclass support.
"""

import yaml
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: list = None
    activation: str = "tanh"
    initialization: str = "xavier_normal"
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64, 64]


@dataclass
class TrainingConfig:
    epochs: int = 10000
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    early_stopping_patience: int = 1000


@dataclass
class SamplingConfig:
    n_collocation: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000
    strategy: str = "lhs"


@dataclass
class EquationConfig:
    equation_type: str = "heat_1d"
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class LossWeights:
    physics: float = 1.0
    boundary: float = 10.0
    initial: float = 10.0
    data: float = 1.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_freq: int = 100
    use_wandb: bool = False
    project_name: str = "pinns-journey"


@dataclass
class Config:
    name: str = "experiment"
    seed: int = 42
    device: str = "auto"
    model: ModelConfig = None
    training: TrainingConfig = None
    sampling: SamplingConfig = None
    equation: EquationConfig = None
    loss_weights: LossWeights = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()
        if self.equation is None:
            self.equation = EquationConfig()
        if self.loss_weights is None:
            self.loss_weights = LossWeights()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        # Handle nested dataclasses
        kwargs = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                if isinstance(value, dict) and hasattr(field.type, '__origin__'):
                    # Handle nested dataclasses
                    if field_name == 'model':
                        kwargs[field_name] = ModelConfig(**value)
                    elif field_name == 'training':
                        kwargs[field_name] = TrainingConfig(**value)
                    elif field_name == 'sampling':
                        kwargs[field_name] = SamplingConfig(**value)
                    elif field_name == 'equation':
                        kwargs[field_name] = EquationConfig(**value)
                    elif field_name == 'loss_weights':
                        kwargs[field_name] = LossWeights(**value)
                    elif field_name == 'logging':
                        kwargs[field_name] = LoggingConfig(**value)
                    else:
                        kwargs[field_name] = value
                else:
                    kwargs[field_name] = value
        return cls(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def create_default_configs(output_dir: str = "configs"):
    """Create default configuration files for each equation type."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Heat equation
    heat = Config(
        name="heat_1d",
        equation=EquationConfig(
            equation_type="heat_1d",
            parameters={"alpha": 0.01}
        ),
        loss_weights=LossWeights(physics=1.0, boundary=10.0, initial=10.0),
        sampling=SamplingConfig(n_collocation=10000, n_boundary=1000, n_initial=1000)
    )
    heat.to_yaml(output / "heat_1d.yaml")
    
    # Burgers equation
    burgers = Config(
        name="burgers_1d",
        equation=EquationConfig(
            equation_type="burgers_1d",
            parameters={"nu": 0.01/3.14159}
        ),
        loss_weights=LossWeights(physics=1.0, boundary=10.0, initial=10.0),
        model=ModelConfig(hidden_layers=[64, 64, 64, 64, 64]),
    )
    burgers.to_yaml(output / "burgers_1d.yaml")
    
    # Wave equation
    wave = Config(
        name="wave_1d",
        equation=EquationConfig(
            equation_type="wave_1d",
            parameters={"c": 1.0}
        ),
        loss_weights=LossWeights(physics=1.0, boundary=10.0, initial=10.0),
    )
    wave.to_yaml(output / "wave_1d.yaml")
    
    # Exponential decay ODE
    decay = Config(
        name="exponential_decay",
        equation=EquationConfig(
            equation_type="exponential_decay",
            parameters={"decay_rate": 1.0}
        ),
        model=ModelConfig(input_dim=1, output_dim=1, hidden_layers=[32, 32, 32]),
        loss_weights=LossWeights(physics=1.0, boundary=0.0, initial=1.0),
        sampling=SamplingConfig(n_collocation=1000, n_boundary=0, n_initial=100),
    )
    decay.to_yaml(output / "exponential_decay.yaml")
    
    print(f"Default configs created in {output}/")


if __name__ == "__main__":
    create_default_configs()