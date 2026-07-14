"""
Tests for PINNs Library

Unit tests for core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pinns.utils.derivatives import gradient, jacobian, hessian, laplacian
from pinns.models import MLP, count_parameters
from pinns.equations import get_equation


class TestDerivatives:
    """Test automatic differentiation utilities."""
    
    def test_gradient_scalar(self):
        """Test gradient of scalar function."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = x[0]**2 + x[1]**3
        
        grad = gradient(y, x)
        expected = torch.tensor([4.0, 27.0])
        
        assert torch.allclose(grad, expected, atol=1e-5)
    
    def test_gradient_batch(self):
        """Test gradient with batch inputs."""
        x = torch.randn(10, 2, requires_grad=True)
        y = (x[:, 0]**2 + x[:, 1]**2).sum()
        
        grad = gradient(y, x)
        
        assert grad.shape == (10, 2)
        expected = 2 * x
        assert torch.allclose(grad, expected, atol=1e-5)
    
    def test_jacobian(self):
        """Test Jacobian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = torch.stack([x[0]**2, x[1]**3, x[0]*x[1]])
        
        J = jacobian(f, x)
        
        expected = torch.tensor([[2.0, 0.0], [0.0, 12.0], [2.0, 1.0]])
        assert torch.allclose(J, expected, atol=1e-5)
    
    def test_hessian(self):
        """Test Hessian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = x[0]**2 * x[1] + x[1]**3
        
        H = hessian(f, x)
        
        expected = torch.tensor([[4.0, 2.0], [2.0, 12.0]])
        assert torch.allclose(H, expected, atol=1e-5)
    
    def test_laplacian(self):
        """Test Laplacian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = x[0]**2 + x[1]**3
        
        lap = laplacian(f, x)
        
        assert torch.allclose(lap, torch.tensor(14.0), atol=1e-5)


class TestModels:
    """Test neural network models."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        x = torch.randn(10, 2)
        y = model(x)
        
        assert y.shape == (10, 1)
    
    def test_mlp_parameters(self):
        """Test parameter counting."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[64, 64])
        n_params = count_parameters(model)
        
        assert n_params > 0
    
    def test_mlp_activations(self):
        """Test different activations."""
        for act in ['tanh', 'swish', 'relu', 'gelu']:
            model = MLP(input_dim=2, output_dim=1, hidden_dims=[32], activation=act)
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)
            assert not torch.isnan(y).any()


class TestEquations:
    """Test equation definitions."""
    
    def test_heat_equation_residual(self):
        """Test heat equation residual computation."""
        equation = get_equation('heat_1d', alpha=0.01)
        
        class ExactModel(torch.nn.Module):
            def forward(self, xt):
                x = xt[:, 0:1]
                t = xt[:, 1:2]
                return torch.exp(-0.01 * np.pi**2 * t) * torch.sin(np.pi * x)
        
        model = ExactModel()
        
        coords = equation.sample_domain(100)
        residual = equation.residual(model, coords)
        
        assert residual.abs().mean() < 1e-3
    
    def test_heat_equation_analytical(self):
        """Test analytical solution."""
        equation = get_equation('heat_1d', alpha=0.01)
        
        coords = {'x': torch.tensor([[0.5]]), 't': torch.tensor([[0.0]])}
        sol = equation.analytical_solution(coords)
        
        assert torch.allclose(sol, torch.tensor([[1.0]]), atol=1e-5)
    
    def test_ode_exponential_decay(self):
        """Test exponential decay ODE."""
        equation = get_equation('exponential_decay', decay_rate=1.0)
        
        class ExactModel(torch.nn.Module):
            def forward(self, t):
                return torch.exp(-t)
        
        model = ExactModel()
        coords = equation.sample_domain(100)
        residual = equation.residual(model, coords)
        
        assert residual.abs().mean() < 1e-4
    
    def test_burgers_equation(self):
        """Test Burgers equation exists and runs."""
        equation = get_equation('burgers_1d', nu=0.01/np.pi)
        
        coords = equation.sample_domain(100)
        assert 'x' in coords and 't' in coords
        
        bc_coords = equation.sample_boundary(50)
        assert 'x' in bc_coords and 't' in bc_coords


class TestSeeding:
    """Test reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test that same seed gives same results."""
        from pinns.utils.seeding import set_seed
        
        set_seed(42)
        model1 = MLP(input_dim=2, output_dim=1, hidden_dims=[32])
        out1 = model1(torch.randn(5, 2))
        
        set_seed(42)
        model2 = MLP(input_dim=2, output_dim=1, hidden_dims=[32])
        out2 = model2(torch.randn(5, 2))
        
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
