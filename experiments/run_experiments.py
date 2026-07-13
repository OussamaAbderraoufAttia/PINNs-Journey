"""
Experiment Runner

Scripts for systematic experiments and ablation studies.
"""

import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from src.pinns import Config, set_seed
from src.pinns.equations import get_equation
from src.pinns.models import MLP
from src.pinns.losses import create_pinn_loss
from src.pinns.training import Trainer, TrainingConfig
from src.pinns.evaluation import evaluate_model


def run_experiment(config_path: str, output_dir: str = "experiments/outputs"):
    """Run a single experiment from config file."""
    config = Config.from_yaml(config_path)
    set_seed(config.seed)
    
    # Create equation
    eq_config = config.equation
    equation = get_equation(
        eq_config.equation_type,
        **eq_config.parameters,
    )
    
    # Create model
    model_config = config.model
    model = MLP(
        input_dim=model_config.input_dim,
        output_dim=model_config.output_dim,
        hidden_dims=model_config.hidden_layers,
        activation=model_config.activation,
    )
    
    # Create loss
    loss_fn = create_pinn_loss(
        equation,
        weights=config.training.loss_weights,
    )
    
    # Create trainer
    trainer = Trainer(
        model,
        equation,
        loss_fn,
        TrainingConfig(
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            optimizer=config.training.optimizer,
            scheduler=config.training.scheduler,
            log_freq=config.logging.log_freq,
            save_freq=config.logging.save_freq,
        ),
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate
    results = evaluate_model(model, equation)
    
    # Save results
    output_path = Path(output_dir) / Path(config_path).stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'config': config.to_dict(),
        'history': history,
        'results': results,
        'model_state': model.state_dict(),
    }, output_path / "results.pt")
    
    return results


def run_ablation_study(base_config: str, param_name: str, values: list, output_dir: str):
    """Run ablation study over a parameter."""
    base = Config.from_yaml(base_config)
    
    results = {}
    for value in values:
        print(f"\nRunning {param_name}={value}")
        config_dict = base.to_dict()
        config_dict[param_name] = value
        
        config = Config.from_dict(config_dict)
        result = run_experiment(config, output_dir)
        results[str(value)] = result
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--output", type=str, default="experiments/outputs")
    parser.add_argument("--ablation", type=str, help="Parameter to ablate")
    parser.add_argument("--values", nargs="+", type=float, help="Values for ablation")
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_study(args.config, args.ablation, args.values, args.output)
    else:
        run_experiment(args.config, args.output)