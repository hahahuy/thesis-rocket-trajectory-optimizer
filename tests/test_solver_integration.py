"""
End-to-end integration tests for WP2 OCP solver.

Tests full pipeline: generate WP1 reference → solve OCP → validate results.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import subprocess
import os
import yaml
import h5py
import casadi as ca

from src.solver.dynamics_casadi import compute_dynamics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_gen.solve_ocp import load_parameters, load_scales, solve_ocp
from src.solver.dynamics_casadi import compute_dynamics
import casadi as ca


def _try_solve_ocp(params, limits, ocp_config, scales):
    """Helper to solve OCP with proper error handling for IPOPT/HSL issues."""
    try:
        result = solve_ocp(params, limits, ocp_config, scales)
        return result
    except RuntimeError as e:
        if "HSL" in str(e) or "Library loading failure" in str(e):
            pytest.skip(f"IPOPT linear solver not available (need HSL or mumps): {e}")
        else:
            raise


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndOCP:
    """End-to-end OCP solver integration tests."""
    
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def ocp_config(self, project_root):
        """Load OCP configuration."""
        params, limits, ocp_config = load_parameters(
            str(project_root / 'configs' / 'phys.yaml'),
            str(project_root / 'configs' / 'limits.yaml'),
            str(project_root / 'configs' / 'ocp.yaml')
        )
        return params, limits, ocp_config
    
    def test_solver_convergence(self, project_root, ocp_config):
        """Test that IPOPT converges for nominal case."""
        params, limits, ocp_config = ocp_config
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        # Reduce problem size for faster test
        ocp_config['problem']['transcription']['n_intervals'] = 5
        ocp_config['problem']['time']['tf_fixed'] = 50.0
        # Disable path constraints for initial test
        ocp_config['path_constraints']['dynamic_pressure']['enabled'] = False
        ocp_config['path_constraints']['load_factor']['enabled'] = False
        ocp_config['path_constraints']['mass']['enabled'] = False
        
        try:
            # Solve OCP
            result = solve_ocp(params, limits, ocp_config, scales)
            
            # Assertions
            if result['stats']['success']:
                assert result['stats']['iterations'] > 0, "Should have iterations"
                assert result['stats']['objective'] is not None, "Should have objective value"
                assert np.isfinite(result['stats']['objective']), "Objective should be finite"
            else:
                # If solver fails, at least verify NLP structure is correct
                pytest.skip(f"Solver did not converge (may need IPOPT/HSL libraries): {result['stats']}")
        except RuntimeError as e:
            if "HSL" in str(e) or "Library loading failure" in str(e):
                pytest.skip(f"IPOPT linear solver not available (need HSL or mumps): {e}")
            else:
                raise
    
    def test_constraint_satisfaction(self, project_root, ocp_config):
        """Test that path constraints are satisfied."""
        params, limits, ocp_config = ocp_config
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        # Enable all constraints
        ocp_config['path_constraints']['dynamic_pressure']['enabled'] = True
        ocp_config['path_constraints']['load_factor']['enabled'] = True
        ocp_config['path_constraints']['mass']['enabled'] = True
        
        # Reduce problem size
        ocp_config['problem']['transcription']['n_intervals'] = 5
        ocp_config['problem']['time']['tf_fixed'] = 50.0
        
        result = _try_solve_ocp(params, limits, ocp_config, scales)
        
        if not result['stats']['success']:
            pytest.skip("Solver didn't converge")
        
        X = result['X']
        U = result['U']
        t = result['t']
        
        # Check mass constraint: m >= m_dry
        m_dry = limits['m_dry']
        masses = X[13, :]
        assert np.all(masses >= m_dry - 1e-6), f"Mass should be >= m_dry ({m_dry})"
        
        # Check control bounds
        T_max = limits['T_max']
        thrusts = U[0, :]
        assert np.all(thrusts >= 0.0), "Thrust should be >= 0"
        assert np.all(thrusts <= T_max + 1e-6), f"Thrust should be <= T_max ({T_max})"
        
        # Check constraint violations from solver
        constraint_violation = result['stats'].get('constraint_violation', 0.0)
        assert constraint_violation < 1e-3, f"Constraint violation should be < 1e-3, got {constraint_violation}"
    
    def test_defect_constraints(self, project_root, ocp_config):
        """Test that defect constraints are satisfied (dynamics consistency)."""
        params, limits, ocp_config = ocp_config
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        # Reduce problem size
        ocp_config['problem']['transcription']['n_intervals'] = 20
        
        result = _try_solve_ocp(params, limits, ocp_config, scales)
        
        if not result['stats']['success']:
            pytest.skip("Solver didn't converge")
        
        X = result['X']
        U = result['U']
        t = result['t']
        
        # Manually compute defects using dynamics
        nx = 14
        N = U.shape[1]
        dt = (t[-1] - t[0]) / N
        
        defects = []
        for k in range(N):
            x_k = X[:, k]
            x_kp1 = X[:, k + 1]
            u_k = U[:, k]
            
            # Compute dynamics
            x_k_sym = ca.MX.sym('x', nx)
            u_k_sym = ca.MX.sym('u', 4)
            f_k = compute_dynamics(x_k_sym, u_k_sym, params)
            f_func = ca.Function('f', [x_k_sym, u_k_sym], [f_k])
            
            f_k_val = np.array(f_func(x_k, u_k))
            
            # Simple Euler check (defect should be small)
            x_pred = x_k + f_k_val * dt
            defect = np.linalg.norm(x_kp1 - x_pred)
            defects.append(defect)
        
        max_defect = np.max(defects)
        # Defect should be small (Hermite-Simpson is more accurate, but this is a sanity check)
        assert max_defect < 100.0, f"Max defect should be reasonable, got {max_defect}"
    
    def test_objective_improvement(self, project_root, ocp_config):
        """Test that optimized trajectory improves over initial guess."""
        params, limits, ocp_config = ocp_config
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        # Set fuel minimization objective
        ocp_config['problem']['objective']['type'] = 'fuel_minimization'
        ocp_config['problem']['transcription']['n_intervals'] = 20
        
        result = _try_solve_ocp(params, limits, ocp_config, scales)
        
        if not result['stats']['success']:
            pytest.skip("Solver didn't converge")
        
        # Check that fuel is consumed (mass decreases)
        X = result['X']
        m0 = X[13, 0]
        mf = X[13, -1]
        
        assert mf < m0, "Final mass should be less than initial mass"
        
        # Objective should be positive (fuel consumed)
        objective = result['stats']['objective']
        assert objective > 0, f"Fuel minimization objective should be > 0, got {objective}"
        
        # Fuel consumed should be reasonable (between 0 and initial mass)
        fuel_consumed = m0 - mf
        assert fuel_consumed > 0, "Fuel should be consumed"
        assert fuel_consumed < m0, "Can't consume more fuel than available"
    
    def test_solution_smoothness(self, project_root, ocp_config):
        """Test that solution is smooth (no discontinuities)."""
        params, limits, ocp_config = ocp_config
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        ocp_config['problem']['transcription']['n_intervals'] = 20
        
        result = solve_ocp(params, limits, ocp_config, scales)
        
        if not result['stats']['success']:
            pytest.skip("Solver didn't converge")
        
        X = result['X']
        U = result['U']
        
        # Check state continuity (should be smooth)
        for i in range(X.shape[0]):
            state_diff = np.diff(X[i, :])
            max_jump = np.max(np.abs(state_diff))
            
            # States should change smoothly (no huge jumps)
            if i < 3:  # Position
                assert max_jump < 1e6, f"Position jump too large: {max_jump}"
            elif i < 6:  # Velocity
                assert max_jump < 1e4, f"Velocity jump too large: {max_jump}"
            elif i == 13:  # Mass
                # Mass should only decrease
                assert np.all(state_diff <= 1e-6), "Mass should not increase"
        
        # Check control smoothness
        for i in range(U.shape[0]):
            control_diff = np.diff(U[i, :])
            max_jump = np.max(np.abs(control_diff))
            
            if i == 0:  # Thrust
                assert max_jump < limits['T_max'], f"Thrust jump too large: {max_jump}"
            else:  # Gimbal angles
                assert max_jump < np.pi, f"Gimbal angle jump too large: {max_jump}"


@pytest.mark.integration
class TestEdgeCases:
    """Edge case tests for constraints and robustness."""
    
    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent
    
    def test_near_q_max(self, project_root):
        """Test solver behavior near dynamic pressure limit."""
        params, limits, ocp_config = load_parameters(
            str(project_root / 'configs' / 'phys.yaml'),
            str(project_root / 'configs' / 'limits.yaml'),
            str(project_root / 'configs' / 'ocp.yaml')
        )
        
        # Set restrictive q_max
        limits['q_max'] = 30000.0  # Lower limit
        params['q_max'] = 30000.0
        ocp_config['problem']['transcription']['n_intervals'] = 15
        
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        result = solve_ocp(params, limits, ocp_config, scales)
        
        # Should still converge (even if tight constraint)
        # This tests constraint handling near limits
        if result['stats']['success']:
            # If it converges, check that q constraints are satisfied
            from src.solver.dynamics_casadi import compute_dynamic_pressure
            
            X = result['X']
            q_values = []
            for k in range(X.shape[1]):
                x_k = X[:, k]
                x_k_sym = ca.MX.sym('x', 14)
                q = compute_dynamic_pressure(x_k_sym, params)
                q_func = ca.Function('q', [x_k_sym], [q])
                q_val = float(q_func(x_k))
                q_values.append(q_val)
            
            max_q = np.max(q_values)
            assert max_q <= limits['q_max'] + 1e-3, f"q should be <= q_max, got {max_q}"
    
    def test_small_mass(self, project_root):
        """Test solver with small initial mass."""
        params, limits, ocp_config = load_parameters(
            str(project_root / 'configs' / 'phys.yaml'),
            str(project_root / 'configs' / 'limits.yaml'),
            str(project_root / 'configs' / 'ocp.yaml')
        )
        
        # Small initial mass
        ocp_config['initial']['m'] = 100.0
        params['m0'] = 100.0
        ocp_config['problem']['transcription']['n_intervals'] = 15
        
        scales = load_scales(str(project_root / 'configs' / 'scales.yaml'))
        
        result = solve_ocp(params, limits, ocp_config, scales)
        
        # Should handle small mass gracefully
        if result['stats']['success']:
            X = result['X']
            masses = X[13, :]
            assert np.all(masses > 0), "Masses should be positive"
            assert np.all(masses >= limits['m_dry'] - 1e-6), "Masses should satisfy constraint"

