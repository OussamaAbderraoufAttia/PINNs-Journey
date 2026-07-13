"""
Automatic Differentiation Utilities

Core utilities for computing derivatives, gradients, Jacobians, Hessians,
and Laplacians using PyTorch's autograd system. These are the building
blocks for PINN loss functions.
"""

import torch
from torch import Tensor
from typing import Tuple, List, Optional, Union


def gradient(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute gradient dy/dx using autograd.
    
    Args:
        y: Output tensor of shape (N, ...) or (N,)
        x: Input tensor of shape (N, D) or (N,), must have requires_grad=True
        create_graph: If True, create computation graph for higher-order derivatives
        retain_graph: If True, retain computation graph
    
    Returns:
        Gradient tensor of shape (N, D) or (N,)
    
    Example:
        >>> x = torch.randn(10, 2, requires_grad=True)
        >>> y = x[:, 0]**2 + x[:, 1]**3
        >>> grad = gradient(y, x)  # Shape: (10, 2)
    """
    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")
    
    # Ensure y is scalar per sample for grad computation
    if y.dim() > 1:
        y = y.sum()
    
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]
    
    return grad


def jacobian(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute Jacobian matrix dy_i/dx_j.
    
    Args:
        y: Output tensor of shape (N, M) or (M,)
        x: Input tensor of shape (N, D) or (D,), must have requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Jacobian tensor of shape (N, M, D) or (M, D)
    
    Example:
        >>> x = torch.randn(10, 2, requires_grad=True)
        >>> y = torch.stack([x[:, 0]**2, x[:, 1]**3], dim=1)
        >>> J = jacobian(y, x)  # Shape: (10, 2, 2)
    """
    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")
    
    N = y.shape[0] if y.dim() > 1 else 1
    M = y.shape[-1] if y.dim() > 1 else y.shape[0]
    D = x.shape[-1] if x.dim() > 1 else x.shape[0]
    
    jac = []
    for i in range(M):
        y_i = y[:, i] if y.dim() > 1 else y[i]
        grad_i = gradient(y_i, x, create_graph=create_graph, retain_graph=retain_graph)
        jac.append(grad_i)
    
    return torch.stack(jac, dim=-2)  # (N, M, D)


def hessian(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute Hessian matrix d²y/dx_i dx_j.
    
    Args:
        y: Scalar output tensor of shape (N,) or ()
        x: Input tensor of shape (N, D) or (D,), must have requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Hessian tensor of shape (N, D, D) or (D, D)
    
    Example:
        >>> x = torch.randn(10, 2, requires_grad=True)
        >>> y = (x[:, 0]**2 + x[:, 1]**2).sum()
        >>> H = hessian(y, x)  # Shape: (10, 2, 2) - diagonal 2s
    """
    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")
    
    # First gradient
    grad_y = gradient(y, x, create_graph=create_graph, retain_graph=retain_graph)
    
    # Second derivatives
    D = grad_y.shape[-1]
    hess = []
    for i in range(D):
        grad_i = grad_y[:, i] if grad_y.dim() > 1 else grad_y[i]
        hess_i = gradient(grad_i, x, create_graph=create_graph, retain_graph=retain_graph)
        hess.append(hess_i)
    
    return torch.stack(hess, dim=-1)  # (N, D, D)


def laplacian(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute Laplacian ∇²y = sum(d²y/dx_i²).
    
    Args:
        y: Scalar output tensor of shape (N,) or ()
        x: Input tensor of shape (N, D) or (D,), must have requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Laplacian tensor of shape (N,) or ()
    
    Example:
        >>> x = torch.randn(10, 2, requires_grad=True)
        >>> y = (x[:, 0]**2 + x[:, 1]**2)
        >>> lap = laplacian(y, x)  # Shape: (10,) - all 4s
    """
    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")
    
    grad_y = gradient(y, x, create_graph=create_graph, retain_graph=retain_graph)
    
    # Divergence of gradient = trace of Hessian
    D = grad_y.shape[-1]
    lap = torch.zeros_like(y)
    for i in range(D):
        grad_i = grad_y[:, i] if grad_y.dim() > 1 else grad_y[i]
        lap_i = gradient(grad_i, x, create_graph=create_graph, retain_graph=retain_graph)
        lap = lap + (lap_i[:, i] if lap_i.dim() > 1 else lap_i[i])
    
    return lap


def divergence(
    v: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute divergence of vector field v(x) = sum(d v_i / d x_i).
    
    Args:
        v: Vector field tensor of shape (N, D) or (D,)
        x: Input coordinates tensor of shape (N, D) or (D,), requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Divergence tensor of shape (N,) or ()
    """
    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")
    
    D = v.shape[-1]
    div = torch.zeros_like(v[..., 0])
    
    for i in range(D):
        v_i = v[:, i] if v.dim() > 1 else v[i]
        div_i = gradient(v_i, x, create_graph=create_graph, retain_graph=retain_graph)
        div = div + (div_i[:, i] if div_i.dim() > 1 else div_i[i])
    
    return div


def curl_2d(
    v: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute 2D curl (scalar) of vector field v = (v_x, v_y).
    curl = dv_y/dx - dv_x/dy
    
    Args:
        v: Vector field tensor of shape (N, 2)
        x: Input coordinates tensor of shape (N, 2), requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Curl tensor of shape (N,)
    """
    if v.shape[-1] != 2 or x.shape[-1] != 2:
        raise ValueError("curl_2d requires 2D vector field and coordinates")
    
    dv_y_dx = gradient(v[:, 1], x, create_graph=create_graph, retain_graph=retain_graph)[:, 0]
    dv_x_dy = gradient(v[:, 0], x, create_graph=create_graph, retain_graph=retain_graph)[:, 1]
    
    return dv_y_dx - dv_x_dy


def curl_3d(
    v: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute 3D curl of vector field v = (v_x, v_y, v_z).
    curl = (dv_z/dy - dv_y/dz, dv_x/dz - dv_z/dx, dv_y/dx - dv_x/dy)
    
    Args:
        v: Vector field tensor of shape (N, 3)
        x: Input coordinates tensor of shape (N, 3), requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Curl tensor of shape (N, 3)
    """
    if v.shape[-1] != 3 or x.shape[-1] != 3:
        raise ValueError("curl_3d requires 3D vector field and coordinates")
    
    curl_x = gradient(v[:, 2], x, create_graph=create_graph, retain_graph=retain_graph)[:, 1] \
             - gradient(v[:, 1], x, create_graph=create_graph, retain_graph=retain_graph)[:, 2]
    curl_y = gradient(v[:, 0], x, create_graph=create_graph, retain_graph=retain_graph)[:, 2] \
             - gradient(v[:, 2], x, create_graph=create_graph, retain_graph=retain_graph)[:, 0]
    curl_z = gradient(v[:, 1], x, create_graph=create_graph, retain_graph=retain_graph)[:, 0] \
             - gradient(v[:, 0], x, create_graph=create_graph, retain_graph=retain_graph)[:, 1]
    
    return torch.stack([curl_x, curl_y, curl_z], dim=-1)


def time_derivative(
    y: Tensor,
    t: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute time derivative dy/dt.
    
    Args:
        y: Output tensor of shape (N, ...) or (N,)
        t: Time tensor of shape (N,) or (N, 1), requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Time derivative tensor of same shape as y
    """
    if t.dim() > 1:
        t = t.squeeze(-1)
    return gradient(y, t, create_graph=create_graph, retain_graph=retain_graph)


def spatial_gradient(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """
    Compute spatial gradient ∇u = (du/dx, du/dy, ...).
    
    Args:
        y: Scalar field of shape (N,) or (N, 1)
        x: Spatial coordinates of shape (N, D), requires_grad=True
        create_graph: If True, create computation graph
        retain_graph: If True, retain computation graph
    
    Returns:
        Gradient tensor of shape (N, D)
    """
    return gradient(y, x, create_graph=create_graph, retain_graph=retain_graph)


def batch_jacobian(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """
    Efficiently compute batched Jacobian using vmap (PyTorch 2.0+).
    
    Args:
        y: Output tensor of shape (N, M)
        x: Input tensor of shape (N, D), requires_grad=True
        create_graph: If True, create computation graph
    
    Returns:
        Jacobian tensor of shape (N, M, D)
    """
    if hasattr(torch, 'vmap'):
        # Use vmap for efficient batched Jacobian
        def get_jac_single(y_i, x_i):
            return torch.autograd.functional.jacobian(
                lambda x: y_i, x_i, create_graph=create_graph
            )
        return torch.vmap(get_jac_single)(y, x)
    else:
        # Fallback to loop
        return jacobian(y, x, create_graph=create_graph)


def batch_hessian(
    y: Tensor,
    x: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """
    Efficiently compute batched Hessian using vmap (PyTorch 2.0+).
    
    Args:
        y: Scalar output tensor of shape (N,)
        x: Input tensor of shape (N, D), requires_grad=True
        create_graph: If True, create computation graph
    
    Returns:
        Hessian tensor of shape (N, D, D)
    """
    if hasattr(torch, 'vmap'):
        def get_hess_single(y_i, x_i):
            return torch.autograd.functional.hessian(
                lambda x: y_i, x_i, create_graph=create_graph
            )
        return torch.vmap(get_hess_single)(y, x)
    else:
        return hessian(y, x, create_graph=create_graph)


class DerivativeTracker:
    """
    Tracks derivative computations for debugging and analysis.
    
    Usage:
        >>> tracker = DerivativeTracker()
        >>> with tracker.track():
        ...     loss = physics_loss(model, x)
        >>> tracker.print_stats()
    """
    
    def __init__(self):
        self.call_counts = {}
        self.tensor_sizes = {}
    
    def track(self):
        return self._TrackingContext(self)
    
    class _TrackingContext:
        def __init__(self, tracker):
            self.tracker = tracker
            self.original_grad = torch.autograd.grad
        
        def __enter__(self):
            torch.autograd.grad = self._wrapped_grad
            return self
        
        def __exit__(self, *args):
            torch.autograd.grad = self.original_grad
        
        def _wrapped_grad(self, *args, **kwargs):
            key = f"grad_{len(args)}"
            self.tracker.call_counts[key] = self.tracker.call_counts.get(key, 0) + 1
            return self.original_grad(*args, **kwargs)
    
    def print_stats(self):
        print("Derivative Computation Stats:")
        for k, v in self.call_counts.items():
            print(f"  {k}: {v} calls")
        self.call_counts.clear()


__all__ = [
    "gradient",
    "jacobian",
    "hessian",
    "laplacian",
    "divergence",
    "curl_2d",
    "curl_3d",
    "time_derivative",
    "spatial_gradient",
    "batch_jacobian",
    "batch_hessian",
    "DerivativeTracker",
]