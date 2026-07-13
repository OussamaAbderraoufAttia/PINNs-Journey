"""
Evaluation Package

Metrics and visualization utilities for PINNs.
"""

from .metrics import (
    l2_error,
    l2_error_abs,
    linf_error,
    relative_error,
    mse,
    mae,
    rmse,
    compute_all_metrics,
    evaluate_model,
    plot_solution_1d,
    plot_solution_2d,
    plot_loss_history,
    plot_residual_heatmap,
    create_comparison_plot,
)

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