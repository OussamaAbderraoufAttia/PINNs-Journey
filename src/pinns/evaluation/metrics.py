"""
Evaluation and Metrics for PINNs

Provides error metrics, visualization utilities, and analysis tools.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def l2_error(pred: Tensor, target: Tensor) -> float:
    """Relative L2 error."""
    return torch.norm(pred - target) / torch.norm(target)


def l2_error_abs(pred: Tensor, target: Tensor) -> float:
    """Absolute L2 error."""
    return torch.norm(pred - target).item()


def linf_error(pred: Tensor, target: Tensor) -> float:
    """L-infinity error (max absolute error)."""
    return torch.max(torch.abs(pred - target)).item()


def relative_error(pred: Tensor, target: Tensor) -> float:
    """Mean relative error."""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8)).item()


def mse(pred: Tensor, target: Tensor) -> float:
    """Mean squared error."""
    return torch.mean((pred - target)**2).item()


def mae(pred: Tensor, target: Tensor) -> float:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred: Tensor, target: Tensor) -> float:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target)**2)).item()


def compute_all_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    """Compute all error metrics."""
    return {
        "l2_relative": l2_error(pred, target),
        "l2_absolute": l2_error_abs(pred, target),
        "linf": linf_error(pred, target),
        "relative": relative_error(pred, target),
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
    }


def evaluate_model(
    model: nn.Module,
    equation: Callable,
    coords: Dict[str, Tensor],
    analytical_solution: Callable = None,
) -> Dict[str, any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained PINN model
        equation: Equation object
        coords: Test coordinates
        analytical_solution: Optional analytical solution function
    
    Returns:
        Dictionary with predictions, errors, and metrics
    """
    model.eval()
    with torch.no_grad():
        x = equation.domain.to_tensor(coords)
        pred = model(x)
        
        results = {
            "predictions": pred.cpu(),
            "coordinates": {k: v.cpu() for k, v in coords.items()},
        }
        
        if analytical_solution is not None:
            target = analytical_solution(coords)
            results["target"] = target.cpu()
            results["metrics"] = compute_all_metrics(pred.cpu(), target.cpu())
        
        # Compute physics residual
        residual = equation.residual(model, coords)
        results["physics_residual"] = residual.cpu()
        results["residual_stats"] = {
            "mean": residual.mean().item(),
            "std": residual.std().item(),
            "max": residual.max().item(),
            "l2": torch.norm(residual).item(),
        }
        
        return results


def plot_solution_1d(
    results: Dict,
    equation: Callable,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """Plot 1D solution comparison."""
    coords = results["coordinates"]
    pred = results["predictions"]
    
    x = coords.get('x', coords.get(list(coords.keys())[0])).numpy().flatten()
    u_pred = pred.numpy().flatten()
    
    fig, axes = plt.subplots(1, 2 if "target" in results else 1, figsize=(12, 4))
    if "target" not in results:
        axes = [axes]
    
    # Prediction
    axes[0].plot(x, u_pred, 'b-', label='PINN', linewidth=2)
    if "target" in results:
        u_true = results["target"].numpy().flatten()
        axes[0].plot(x, u_true, 'r--', label='Exact', linewidth=2)
        axes[0].set_title(f"Solution (L2 rel: {results['metrics']['l2_relative']:.2e})")
    else:
        axes[0].set_title("PINN Prediction")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error
    if "target" in results:
        error = np.abs(u_pred - u_true)
        axes[1].semilogy(x, error, 'g-', linewidth=2)
        axes[1].set_title("Absolute Error")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('|Error|')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_solution_2d(
    results: Dict,
    equation: Callable,
    time_idx: int = -1,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """Plot 2D solution as heatmap."""
    coords = results["coordinates"]
    pred = results["predictions"]
    
    # Assume grid-like data for 2D
    x = coords['x'].numpy()
    y = coords['y'].numpy()
    u = pred.numpy()
    
    # Reshape if needed
    if len(x.shape) == 1:
        # Try to infer grid
        n_unique_x = len(np.unique(x))
        n_unique_y = len(np.unique(y))
        if n_unique_x * n_unique_y == len(x):
            x = x.reshape(n_unique_y, n_unique_x)
            y = y.reshape(n_unique_y, n_unique_x)
            u = u.reshape(n_unique_y, n_unique_x)
    
    fig, axes = plt.subplots(1, 3 if "target" in results else 2, figsize=(15, 4))
    if "target" not in results:
        axes = [axes[0], axes[1]]
    
    # Prediction
    im1 = axes[0].contourf(x, y, u, levels=50, cmap='RdBu_r')
    axes[0].set_title("PINN Prediction")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # True solution
    if "target" in results:
        u_true = results["target"].numpy()
        if len(u_true.shape) == 1:
            u_true = u_true.reshape(n_unique_y, n_unique_x)
        
        im2 = axes[1].contourf(x, y, u_true, levels=50, cmap='RdBu_r')
        axes[1].set_title("Exact Solution")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Error
        error = np.abs(u - u_true)
        im3 = axes[2].contourf(x, y, error, levels=50, cmap='hot')
        axes[2].set_title(f"Error (L2: {results['metrics']['l2_relative']:.2e})")
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_loss_history(
    history: Dict[str, List[float]],
    save_path: str = None,
    show: bool = True,
    log_scale: bool = True,
) -> plt.Figure:
    """Plot training loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = history["epoch"]
    losses = history["losses"]
    
    for name, values in losses.items():
        ax.plot(epochs, values, label=name, linewidth=1.5)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_residual_heatmap(
    results: Dict,
    equation: Callable,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """Plot PDE residual as heatmap."""
    coords = results["coordinates"]
    residual = results["physics_residual"].numpy()
    
    # For time-dependent, pick a time slice
    if 't' in coords:
        t = coords['t'].numpy().flatten()
        unique_t = np.unique(t)
        if len(unique_t) > 1:
            # Plot at multiple times
            n_times = min(4, len(unique_t))
            fig, axes = plt.subplots(1, n_times, figsize=(4*n_times, 4))
            if n_times == 1:
                axes = [axes]
            
            for i, t_val in enumerate(unique_t[:n_times]):
                mask = np.abs(t - t_val) < 1e-6
                x = coords['x'].numpy().flatten()[mask]
                r = residual.flatten()[mask]
                
                axes[i].scatter(x, r, s=1, alpha=0.5)
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('Residual')
                axes[i].set_title(f't = {t_val:.2f}')
                axes[i].grid(True, alpha=0.3)
        else:
            # Single time
            fig, ax = plt.subplots(figsize=(8, 4))
            x = coords['x'].numpy().flatten()
            ax.scatter(x, residual.flatten(), s=1, alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('Residual')
            ax.set_title('PDE Residual')
            ax.grid(True, alpha=0.3)
    else:
        # Steady state
        fig, ax = plt.subplots(figsize=(8, 4))
        x = coords[list(coords.keys())[0]].numpy().flatten()
        ax.scatter(x, residual.flatten(), s=1, alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title('PDE Residual')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_comparison_plot(
    models: Dict[str, nn.Module],
    equation: Callable,
    test_coords: Dict[str, Tensor],
    analytical_solution: Callable,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """Create comparison plot for multiple models."""
    results = {}
    
    for name, model in models.items():
        results[name] = evaluate_model(model, equation, test_coords, analytical_solution)
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    coords = test_coords
    x = coords['x'].numpy().flatten()
    
    # Get analytical solution
    u_true = analytical_solution(coords).numpy().flatten()
    
    for i, (name, res) in enumerate(results.items()):
        u_pred = res["predictions"].numpy().flatten()
        
        # Solution
        axes[0, i].plot(x, u_true, 'k--', label='Exact', linewidth=2)
        axes[0, i].plot(x, u_pred, 'b-', label=name, linewidth=2)
        axes[0, i].set_title(f"{name} (L2: {res['metrics']['l2_relative']:.2e})")
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('u')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Error
        error = np.abs(u_pred - u_true)
        axes[1, i].semilogy(x, error, 'r-', linewidth=2)
        axes[1, i].set_title(f"Absolute Error")
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('|Error|')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


__all__ = [
    "l2_error",
    "l2_error_abs",
    "linf_error",
    "relative_error",
    "mse",
    "mae",
    "rmse",
    "compute_all_metrics",
    "evaluate_model",
    "plot_solution_1d",
    "plot_solution_2d",
    "plot_loss_history",
    "plot_residual_heatmap",
    "create_comparison_plot",
]