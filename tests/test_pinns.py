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
        expected = torch.tensor(14.0)
        assert torch.allclose(lap, expected, atol=1e-5)


class TestModels:
    """Test neural network models."""
    
    def test_mlp_creation(self):
        """Test MLP creation."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        assert isinstance(model, MLP)
        assert model.input_dim == 2
        assert model.output_dim == 1
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        x = torch.randn(10, 2)
        y = model(x)
        assert y.shape == (10, 1)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = MLP(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        count = count_parameters(model)
        # 2*32 + 32 + 32*32 + 32 + 32*1 + 1 = 64+32+1024+32+32+1 = 1185
        assert count == 1185
    
    def test_different_activations(self):
        """Test different activation functions."""
        for act in ['tanh', 'sin', 'swish', 'gelu', 'relu']:
            model = MLP(input_dim=2, output_dim=1, hidden_dims=[16], activation=act)
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)
    
    def test_initializers(self):
        """Test different weight initializations."""
        for init in ['xavier_normal', 'xavier_uniform', 'he_normal', 'orthogonal']:
            model = MLP(input_dim=2, output_dim=1, hidden_dims=[16], init_type=init)
            x = torch.randn(5, 2)
            y = model(x)
            assert y.shape == (5, 1)


class TestEquations:
    """Test equation implementations."""
    
    def test_heat_equation_creation(self):
        """Test heat equation creation."""
        eq = get_equation('heat_1d', alpha=0.01)
        assert eq is not None
        assert eq.alpha == 0.01
    
    def test_heat_equation_residual(self):
        """Test heat equation residual computation."""
        eq = get_equation('heat_1d', alpha=0.01)
        
        # Simple linear function: u = x
        # u_t = 0, u_xx = 0, residual = 0
        def model(x):
            return x[:, 0:1]  # u = x
        
        coords = {'x': torch.tensor([[0.5]], requires_grad=True), 
                  't': torch.tensor([[0.1]], requires_grad=True)}
        
        residual = eq.residual(model, coords)
        assert torch.allclose(residual, torch.tensor(0.0), atol=1e-5)
    
    def test_heat_analytical_solution(self):
        """Test heat equation analytical solution."""
        eq = get_equation('heat_1d', alpha=0.01)
        
        coords = {'x': torch.tensor([[0.5]]), 't': torch.tensor([[0.0]])}
        u_exact = eq.analytical_solution(coords)
        
        # u(x,0) = sin(pi*x) = sin(pi/2) = 1
        assert torch.allclose(u_exact, torch.tensor([[1.0]]), atol=1e-5)
    
    def test_burgers_equation_creation(self):
        """Test Burgers equation creation."""
        eq = get_equation('burgers_1d', nu=0.01/3.14159)
        assert eq is not None
        assert eq.nu == 0.01/3.14159
    
    def test_equation_registry(self):
        """Test equation registry."""
        from src.pinns.equations import list_equations
        equations = list_equations()
        assert 'heat' in equations
        assert 'heat_1d' in equations
        assert 'burgers' in equations


class TestTraining:
    """Test training components."""
    
    def test_loss_creation(self):
        """Test loss function creation."""
        from src.pinns.losses import create_pinn_loss
        from src.pinns.equations import get_equation
        
        eq = get_equation('heat_1d', alpha=0.01)
        loss_fn = create_pinn_loss(eq)
        
        # Should return a callable
        assert callable(loss_fn)
    
    def test_training_config(self):
        """Test training config creation."""
        from src.pinns.training import TrainingConfig
        
        config = TrainingConfig(
            epochs=1000,
            learning_rate=1e-3,
            optimizer='adam',
        )
        
        assert config.epochs == 1000
        assert config.learning_rate == 1e-3


# Run tests with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])