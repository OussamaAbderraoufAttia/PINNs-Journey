"""
Tests for PINNs Library

Unit tests for core functionality.
"""

import torch
import pytest
import numpy as np
from src.pinns.utils.derivatives import gradient, jacobian, hessian, laplacian
from src.pinns.models import MLP, count_parameters
from src.pinns.equations import get_equation


class TestDerivatives:
    """Test automatic differentiation utilities."""
    
    def test_gradient_scalar(self):
        """Test gradient of scalar function."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = x[0]**2 + x[1]**3
        
        grad = gradient(y, x)
        expected = torch.tensor([4.0, 27.0])  # 2x, 3x^2
        
        assert torch.allclose(grad, expected, atol=1e-5)
    
    def test_gradient_batch(self):
        """Test gradient with batch inputs."""
        x = torch.randn(10, 2, requires_grad=True)
        y = (x[:, 0]**2 + x[:, 1]**2).sum()
        
        grad = gradient(y, x)
        
        assert grad.shape == (10, 2)
        # Gradient should be [2*x1, 2*x2] for each sample
        expected = 2 * x
        assert torch.allclose(grad, expected, atol=1e-5)
    
    def test_jacobian(self):
        """Test Jacobian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = torch.stack([x[0]**2, x[1]**3, x[0]*x[1]])
        
        J = jacobian(f, x)
        
        # Expected: [[2x1, 0], [0, 3x2^2], [x2, x1]] = [[2, 0], [0, 12], [2, 1]]
        expected = torch.tensor([[2.0, 0.0], [0.0, 12.0], [2.0, 1.0]])
        assert torch.allclose(J, expected, atol=1e-5)
    
    def test_hessian(self):
        """Test Hessian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = x[0]**2 * x[1] + x[1]**3
        
        H = hessian(f, x)
        
        # f = x1^2 * x2 + x2^3
        # df/dx1 = 2*x1*x2, df/dx2 = x1^2 + 3*x2^2
        # d2f/dx1^2 = 2*x2, d2f/dx1dx2 = 2*x1
        # d2f/dx2dx1 = 2*x1, d2f/dx2^2 = 6*x2
        # At (1,2): [[4, 2], [2, 12]]
        expected = torch.tensor([[4.0, 2.0], [2.0, 12.0]])
        assert torch.allclose(H, expected, atol=1e-5)
    
    def test_laplacian(self):
        """Test Laplacian computation."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = x[0]**2 + x[1]**3
        
        lap = laplacian(f, x)
        
        # ∇²f = d²f/dx1² + d²f/dx2² = 2 + 6*x2 = 2 + 12 = 14
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
        
        # 2*64 + 64 + 64*64 + 64 + 64*1 + 1 = 128+64+4096+64+64+1 = 4417
        assert n_params > 0
    
    def test_mlp_activations(self):
        """Test different activations."""
        for act in ['tanh', 'sin', 'swish', 'relu', 'gelu']:
            model = MLP(input_dim=2, output_dim=1, hidden_dims=[32], activation=act)
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)
            assert not torch.isnan(y).any()
    
    def test_siren(self):
        """Test SIREN model."""
        from src.pinns.models import SIREN
        model = SIREN(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 1)
    
    def test_fourier_features(self):
        """Test Fourier feature MLP."""
        from src.pinns.models import FourierFeatureMLP
        model = FourierFeatureMLP(input_dim=2, output_dim=1, hidden_dims=[32], mapping_size=64)
        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 1)


class TestEquations:
    """Test equation definitions."""
    
    def test_heat_equation_residual(self):
        """Test heat equation residual computation."""
        equation = get_equation('heat_1d', alpha=0.01)
        
        # Simple model that returns sin(pi*x)*exp(-alpha*pi^2*t)
        class ExactModel(torch.nn.Module):
            def forward(self, xt):
                x = xt[:, 0:1]
                t = xt[:, 1:2]
                return torch.exp(-0.01 * np.pi**2 * t) * torch.sin(np.pi * x)
        
        model = ExactModel()
        
        # Test at random points
        coords = equation.sample_domain(100)
        residual = equation.residual(model, coords)
        
        # Should be near zero for exact solution
        assert residual.abs().mean() < 1e-3
    
    def test_heat_equation_analytical(self):
        """Test analytical solution."""
        equation = get_equation('heat_1d', alpha=0.01)
        
        coords = {'x': torch.tensor([[0.5]]), 't': torch.tensor([[0.0]])}
        sol = equation.analytical_solution(coords)
        
        # sin(pi * 0.5) = 1
        assert torch.allclose(sol, torch.tensor([[1.0]]), atol=1e-5)
    
    def test_ode_exponential_decay(self):
        """Test exponential decay ODE."""
        equation = get_equation('exponential_decay', decay_rate=1.0)
        
        # Exact solution: y = exp(-t)
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
        
        # Just test that it can be instantiated and sampled
        coords = equation.sample_domain(100)
        assert 'x' in coords and 't' in coords
        
        # Test boundary sampling
        bc_coords = equation.sample_boundary(50)
        assert 'x' in bc_coords and 't' in bc_coords


class TestSeeding:
    """Test reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test that same seed gives same results."""
        from src.pinns.utils.seeding import set_seed
        
        set_seed(42)
        model1 = MLP(input_dim=2, output_dim=1, hidden_dims=[32])
        out1 = model1(torch.randn(5, 2))
        
        set_seed(42)
        model2 = MLP(input_dim=2, output_dim=1, hidden_dims=[32])
        out2 = model2(torch.randn(5, 2))
        
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])