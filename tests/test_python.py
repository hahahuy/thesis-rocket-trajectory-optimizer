"""
Python tests for the thesis rocket trajectory optimizer.
"""
import pytest
import numpy as np
import torch
import casadi as ca
import hydra
from omegaconf import DictConfig


class TestBasicFunctionality:
    """Test basic functionality of core dependencies."""
    
    def test_numpy_basic(self):
        """Test basic NumPy operations."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        assert len(arr) == 5
    
    def test_torch_basic(self):
        """Test basic PyTorch operations."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        z = x + y
        assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))
        assert z.sum().item() == 21.0
    
    def test_casadi_basic(self):
        """Test basic CasADi operations."""
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        
        f = x**2 + y**2
        assert f is not None
        
        # Test function evaluation
        f_func = ca.Function('f', [x, y], [f])
        result = f_func(2.0, 3.0)
        assert abs(result - 13.0) < 1e-10
    
    def test_hydra_basic(self):
        """Test basic Hydra configuration."""
        with hydra.initialize(config_path=None):
            cfg = hydra.compose(config_name="config")
            assert isinstance(cfg, DictConfig)


class TestPhysicsModels:
    """Test physics-related functionality."""
    
    def test_gravity_calculation(self):
        """Test gravity calculation."""
        g = 9.81  # m/s^2
        mass = 1000.0  # kg
        force = mass * g
        
        assert abs(force - 9810.0) < 1e-10
    
    def test_trajectory_basic(self):
        """Test basic trajectory calculations."""
        # Simple projectile motion
        v0 = 100.0  # initial velocity m/s
        angle = 45.0  # degrees
        g = 9.81
        
        v0x = v0 * np.cos(np.radians(angle))
        v0y = v0 * np.sin(np.radians(angle))
        
        # Time to peak
        t_peak = v0y / g
        
        # Maximum height
        h_max = v0y**2 / (2 * g)
        
        assert v0x > 0
        assert v0y > 0
        assert t_peak > 0
        assert h_max > 0
    
    def test_torch_autograd(self):
        """Test PyTorch automatic differentiation."""
        x = torch.tensor(2.0, requires_grad=True)
        y = x**3 + 2*x**2 + 3*x + 1
        
        y.backward()
        
        # dy/dx = 3x^2 + 4x + 3
        expected_grad = 3*2**2 + 4*2 + 3  # = 12 + 8 + 3 = 23
        assert abs(x.grad.item() - expected_grad) < 1e-6


class TestOptimization:
    """Test optimization-related functionality."""
    
    def test_casadi_optimization(self):
        """Test basic CasADi optimization."""
        # Simple quadratic optimization
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        
        # Objective: minimize x^2 + y^2
        f = x**2 + y**2
        
        # Constraint: x + y = 1
        g = x + y - 1
        
        # Create NLP
        nlp = {'x': ca.vertcat(x, y), 'f': f, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', nlp)
        
        # Solve
        result = solver(x0=[0, 0], lbg=0, ubg=0)
        
        # Check solution
        x_opt = result['x'][0]
        y_opt = result['x'][1]
        
        # Should be x = y = 0.5
        assert abs(x_opt - 0.5) < 1e-6
        assert abs(y_opt - 0.5) < 1e-6
    
    def test_torch_optimization(self):
        """Test PyTorch optimization."""
        # Simple linear regression
        x = torch.randn(100, 1)
        y = 2 * x + 1 + 0.1 * torch.randn(100, 1)
        
        # Model: y = w * x + b
        w = torch.randn(1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        
        optimizer = torch.optim.Adam([w, b], lr=0.01)
        
        for _ in range(100):
            optimizer.zero_grad()
            y_pred = w * x + b
            loss = torch.mean((y_pred - y)**2)
            loss.backward()
            optimizer.step()
        
        # Check if we learned reasonable parameters
        assert abs(w.item() - 2.0) < 0.5  # Should be close to 2
        assert abs(b.item() - 1.0) < 0.5  # Should be close to 1


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""
    
    def test_large_array_operations(self):
        """Test operations on large arrays."""
        n = 10000
        a = np.random.randn(n)
        b = np.random.randn(n)
        
        # This should be fast
        c = a + b
        assert len(c) == n
    
    def test_torch_large_tensor(self):
        """Test operations on large tensors."""
        n = 1000
        x = torch.randn(n, n)
        y = torch.randn(n, n)
        
        # Matrix multiplication
        z = torch.mm(x, y)
        assert z.shape == (n, n)


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_simulation(self):
        """Test a simplified version of the full pipeline."""
        # This would test the integration of multiple components
        # For now, just test that we can import everything
        import numpy as np
        import torch
        import casadi as ca
        
        # Create a simple trajectory
        t = np.linspace(0, 10, 100)
        x = 100 * t
        y = 50 * t - 0.5 * 9.81 * t**2
        
        # Convert to PyTorch
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)
        
        # Convert to CasADi
        t_sym = ca.MX.sym('t')
        x_sym = 100 * t_sym
        y_sym = 50 * t_sym - 0.5 * 9.81 * t_sym**2
        
        # Test that all representations work
        assert len(x_torch) == 100
        assert len(y_torch) == 100
        assert x_sym is not None
        assert y_sym is not None


if __name__ == "__main__":
    pytest.main([__file__])
