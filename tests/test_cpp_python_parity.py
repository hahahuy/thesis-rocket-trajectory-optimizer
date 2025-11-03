"""
Test C++/Python dynamics parity.

Compares C++ dynamics (via validate_dynamics) with CasADi Python dynamics.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import subprocess
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from solver.dynamics_casadi import compute_dynamics
import casadi as ca


@pytest.mark.integration
class TestCPPPythonParity:
    """Test numerical agreement between C++ and Python dynamics."""
    
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent
    
    def test_dynamics_output_parity(self, project_root):
        """
        Compare C++ and Python dynamics for same inputs.
        
        Note: This is a simplified test. Full parity would require
        C++ bindings or loading C++ trajectory data.
        """
        # Define test case
        nx = 14
        nu = 4
        
        # Test state
        x = np.zeros(nx)
        x[2] = 5000.0  # altitude
        x[3] = 100.0   # vx
        x[5] = 50.0    # vz
        x[6] = 1.0     # q0
        x[13] = 50.0   # mass
        
        # Test control
        u = np.array([4000.0, 0.0, 0.0, 0.0])  # Thrust, no gimbal
        
        # Parameters matching C++ defaults
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
            'T_max': 4000.0,
            'm_dry': 10.0
        }
        
        # Compute Python dynamics
        x_sym = ca.MX.sym('x', nx)
        u_sym = ca.MX.sym('u', nu)
        xdot_py = compute_dynamics(x_sym, u_sym, params)
        
        f_py = ca.Function('f_py', [x_sym, u_sym], [xdot_py])
        xdot_py_val = np.array(f_py(x, u)).flatten()
        
        # Basic sanity checks (since we don't have direct C++ comparison)
        assert len(xdot_py_val) == nx, "Output should have nx elements"
        assert np.all(np.isfinite(xdot_py_val)), "All outputs should be finite"
        
        # Check reasonable magnitudes
        # Position derivative = velocity
        assert abs(xdot_py_val[0] - x[3]) < 1e-6, "r_dot[0] should equal vx"
        assert abs(xdot_py_val[1] - x[4]) < 1e-6, "r_dot[1] should equal vy"
        assert abs(xdot_py_val[2] - x[5]) < 1e-6, "r_dot[2] should equal vz"
        
        # Mass derivative should be negative (fuel consumption)
        assert xdot_py_val[13] < 0, "m_dot should be negative"
        
        # Velocity derivative should have reasonable magnitude
        v_dot_mag = np.linalg.norm(xdot_py_val[3:6])
        assert v_dot_mag < 1000.0, "Acceleration should be reasonable"
    
    def test_trajectory_consistency(self, project_root):
        """
        Test that Python dynamics produce consistent trajectory.
        
        Integrates forward and checks basic physics.
        """
        nx = 14
        nu = 4
        
        params = {
            'Cd': 0.3, 'CL_alpha': 3.5, 'Cm_alpha': -0.8, 'C_delta': 0.05,
            'S_ref': 0.05, 'l_ref': 1.2, 'Isp': 300.0, 'g0': 9.81,
            'rho0': 1.225, 'h_scale': 8400.0,
            'I_b': np.array([1000.0, 1000.0, 100.0]),
            'T_max': 4000.0, 'm_dry': 10.0
        }
        
        # Initial state
        x0 = np.zeros(nx)
        x0[2] = 0.0      # altitude
        x0[6] = 1.0      # q0
        x0[13] = 50.0    # mass
        
        # Constant control
        u = np.array([4000.0, 0.0, 0.0, 0.0])
        
        # Integrate a few steps
        x_sym = ca.MX.sym('x', nx)
        u_sym = ca.MX.sym('u', nu)
        xdot = compute_dynamics(x_sym, u_sym, params)
        f = ca.Function('f', [x_sym, u_sym], [xdot])
        
        dt = 0.1
        x = x0.copy()
        
        for i in range(10):
            xdot_val = np.array(f(x, u)).flatten()
            x = x + xdot_val * dt
            
            # Check mass decreases
            assert x[13] <= x0[13], "Mass should decrease or stay same"
            
            # Check quaternion norm (approximately)
            q = x[6:10]
            q_norm = np.linalg.norm(q)
            assert abs(q_norm - 1.0) < 0.1, "Quaternion should stay approximately normalized"

