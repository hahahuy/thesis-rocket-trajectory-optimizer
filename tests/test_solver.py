"""
Unit and integration tests for OCP solver.

Tests collocation, constraints, initial guess, and solver integration.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import casadi as ca

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from solver.dynamics_casadi import compute_dynamics, compute_dynamic_pressure, compute_load_factor
from solver.collocation import compute_hermite_simpson_step
from solver.transcription import DirectCollocation
from solver.constraints import (
    compute_dynamic_pressure_constraint,
    compute_load_factor_constraint,
    create_state_bounds,
    create_control_bounds
)
from solver.utils import generate_vertical_ascent_guess


class TestDynamicsCasadi:
    """Test CasADi dynamics implementation."""
    
    def test_dynamics_shape(self):
        """Test that dynamics returns correct shape."""
        nx = 14
        nu = 4
        
        x = ca.MX.sym('x', nx)
        u = ca.MX.sym('u', nu)
        
        params = {
            'Cd': 0.3,
            'CL_alpha': 3.5,
            'Cm_alpha': -0.8,
            'C_delta': 0.05,
            'S_ref': 0.05,
            'l_ref': 1.2,
            'Isp': 300.0,
            'g0': 9.81,
            'rho0': 1.225,
            'h_scale': 8400.0,
            'I_b': np.array([1000.0, 1000.0, 100.0]),
            'T_max': 1000000.0,
            'm_dry': 1000.0
        }
        
        xdot = compute_dynamics(x, u, params)
        
        assert xdot.size1() == nx
        assert xdot.size2() == 1
    
    def test_dynamic_pressure(self):
        """Test dynamic pressure computation."""
        nx = 14
        x = ca.MX.sym('x', nx)
        
        params = {
            'rho0': 1.225,
            'h_scale': 8400.0
        }
        
        q = compute_dynamic_pressure(x, params)
        
        assert q.size1() == 1
        assert q.size2() == 1


class TestCollocation:
    """Test collocation methods."""
    
    def test_hermite_simpson_defect(self):
        """Test Hermite-Simpson defect computation."""
        nx = 14
        nu = 4
        
        x_k = ca.MX.sym('x_k', nx)
        u_k = ca.MX.sym('u_k', nu)
        x_kp1 = ca.MX.sym('x_kp1', nx)
        u_kp1 = ca.MX.sym('u_kp1', nu)
        dt = ca.MX.sym('dt')
        
        params = {
            'Cd': 0.3,
            'CL_alpha': 3.5,
            'Cm_alpha': -0.8,
            'C_delta': 0.05,
            'S_ref': 0.05,
            'l_ref': 1.2,
            'Isp': 300.0,
            'g0': 9.81,
            'rho0': 1.225,
            'h_scale': 8400.0,
            'I_b': np.array([1000.0, 1000.0, 100.0]),
            'T_max': 1000000.0,
            'm_dry': 1000.0
        }
        
        f = lambda x, u, p: compute_dynamics(x, u, p)
        
        defect = compute_hermite_simpson_step(f, x_k, u_k, x_kp1, u_kp1, dt, params)
        
        assert defect.size1() == nx
        assert defect.size2() == 1


class TestConstraints:
    """Test constraint functions."""
    
    def test_state_bounds(self):
        """Test state bounds creation."""
        nx = 14
        lbx, ubx = create_state_bounds(nx)
        
        assert len(lbx) == nx
        assert len(ubx) == nx
        assert np.all(lbx <= ubx)
    
    def test_control_bounds(self):
        """Test control bounds creation."""
        nu = 4
        limits = {
            'T_max': 1000000.0,
            'theta_max': 0.1745,
            'phi_max': 0.1745,
            'delta_max': 0.1745
        }
        
        lbu, ubu = create_control_bounds(nu, limits)
        
        assert len(lbu) == nu
        assert len(ubu) == nu
        assert np.all(lbu <= ubu)
        assert lbu[0] == 0.0  # Thrust lower bound
        assert ubu[0] == limits['T_max']


class TestTranscription:
    """Test direct collocation transcription."""
    
    def test_transcription_creation(self):
        """Test that transcription can be created."""
        nx = 14
        nu = 4
        N = 10
        
        params = {
            'Cd': 0.3,
            'CL_alpha': 3.5,
            'Cm_alpha': -0.8,
            'C_delta': 0.05,
            'S_ref': 0.05,
            'l_ref': 1.2,
            'Isp': 300.0,
            'g0': 9.81,
            'rho0': 1.225,
            'h_scale': 8400.0,
            'I_b': np.array([1000.0, 1000.0, 100.0]),
            'T_max': 1000000.0,
            'm_dry': 1000.0,
            'q_max': 40000.0,
            'n_max': 5.0
        }
        
        transcription = DirectCollocation(nx, nu, N, params)
        
        assert transcription.nx == nx
        assert transcription.nu == nu
        assert transcription.N == N
    
    def test_nlp_creation(self):
        """Test NLP creation."""
        nx = 14
        nu = 4
        N = 10
        
        params = {
            'Cd': 0.3,
            'CL_alpha': 3.5,
            'Cm_alpha': -0.8,
            'C_delta': 0.05,
            'S_ref': 0.05,
            'l_ref': 1.2,
            'Isp': 300.0,
            'g0': 9.81,
            'rho0': 1.225,
            'h_scale': 8400.0,
            'I_b': np.array([1000.0, 1000.0, 100.0]),
            'T_max': 1000000.0,
            'm_dry': 1000.0,
            'q_max': 40000.0,
            'n_max': 5.0
        }
        
        transcription = DirectCollocation(nx, nu, N, params)
        
        constraint_types = {
            'dynamic_pressure': True,
            'load_factor': True,
            'mass': True
        }
        
        nlp = transcription.create_nlp(
            objective_type='fuel_minimization',
            constraint_types=constraint_types,
            tf_free=False
        )
        
        assert 'x' in nlp
        assert 'f' in nlp
        assert 'g' in nlp


class TestInitialGuess:
    """Test initial guess generation."""
    
    def test_vertical_ascent_guess(self):
        """Test vertical ascent initial guess."""
        nx = 14
        nu = 4
        N = 20
        
        params = {
            'g0': 9.81,
            'Isp': 300.0,
            'm_dry': 1000.0,
            'T_max': 1000000.0
        }
        
        initial_conditions = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'q0': 1.0,
            'q1': 0.0,
            'q2': 0.0,
            'q3': 0.0,
            'wx': 0.0,
            'wy': 0.0,
            'wz': 0.0,
            'm': 5000.0
        }
        
        config = {
            'tf_fixed': 100.0,
            'tf_min': 30.0,
            'tf_max': 120.0,
            'vertical_ascent': {}
        }
        
        X0, U0, tf_guess = generate_vertical_ascent_guess(
            nx, nu, N, params, initial_conditions, config
        )
        
        assert X0.shape == (nx, N + 1)
        assert U0.shape == (nu, N)
        assert tf_guess > 0.0
        
        # Check initial state
        assert X0[2, 0] == 0.0  # Initial altitude
        assert X0[13, 0] == 5000.0  # Initial mass
        
        # Check quaternion normalization
        q = X0[6:10, :]
        q_norms = np.linalg.norm(q, axis=0)
        assert np.allclose(q_norms, 1.0, atol=1e-6)


@pytest.mark.slow
class TestSolverIntegration:
    """Integration tests for full solver (marked as slow)."""
    
    def test_small_problem(self):
        """Test solving a small OCP problem."""
        # This test would require a full solver run
        # Skip for now or mark as integration test
        pass

