"""
Main OCP solver for rocket trajectory optimization.

Solves the optimal control problem using direct collocation and IPOPT.
"""

import sys
import os
import yaml
import numpy as np
import casadi as ca
from pathlib import Path
from typing import Dict, Tuple, Optional
import h5py
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.solver.transcription import DirectCollocation
from src.solver.utils import generate_initial_guess
from src.solver.constraints import create_state_bounds, create_control_bounds, create_constraint_bounds


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_parameters(phys_path: str, limits_path: str, ocp_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load physical parameters, limits, and OCP configuration."""
    phys_config = load_config(phys_path)
    limits_config = load_config(limits_path)
    ocp_config = load_config(ocp_path)
    
    # Merge physical parameters
    phys = phys_config.get('aerodynamics', {})
    phys.update(phys_config.get('inertia', {}))
    phys.update(phys_config.get('propulsion', {}))
    phys.update(phys_config.get('atmosphere', {}))
    
    # Inertia tensor
    if 'I_b' in phys:
        I_b_list = phys['I_b']
        if isinstance(I_b_list, list) and len(I_b_list) == 9:
            # Full matrix
            I_b = np.array(I_b_list).reshape(3, 3)
        elif isinstance(I_b_list, list) and len(I_b_list) == 3:
            # Diagonal
            I_b = np.array(I_b_list)
        else:
            I_b = np.array([1000.0, 1000.0, 100.0])  # Default
        phys['I_b'] = I_b
    
    # Limits
    limits = {}
    limits['T_max'] = limits_config.get('thrust', {}).get('T_max', 1000000.0)
    limits['m_dry'] = limits_config.get('mass', {}).get('m_dry', 1000.0)
    limits['q_max'] = limits_config.get('aerodynamics', {}).get('q_max', 40000.0)
    limits['n_max'] = limits_config.get('structural', {}).get('n_max', 5.0)
    limits['alpha_max'] = limits_config.get('aerodynamics', {}).get('alpha_max', 0.1)
    limits['theta_max'] = np.deg2rad(limits_config.get('guidance', {}).get('w_gimbal_max', 1.0) * 10.0)  # Approximate
    limits['phi_max'] = limits['theta_max']
    delta_limit_deg = phys_config.get('aerodynamics', {}).get('delta_limit_deg', 10.0)
    limits['delta_max'] = np.deg2rad(delta_limit_deg)
    
    # Add limits to params
    params = phys.copy()
    params.update(limits)
    
    return params, limits, ocp_config


def load_scales(scales_path: str) -> Dict:
    """Load scaling factors."""
    scales_config = load_config(scales_path)
    scales = {}
    
    L_ref = float(scales_config.get('length', {}).get('L_ref', 1e4))
    T_ref = float(scales_config.get('time', {}).get('T_ref', 50.0))
    M_ref = float(scales_config.get('mass', {}).get('M_ref', 50.0))
    V_ref = float(scales_config.get('velocity', {}).get('V_ref', 1e3))
    F_ref = float(scales_config.get('force', {}).get('F_ref', 5e3))
    W_ref = 1.0 / T_ref  # Angular velocity reference
    
    # State scaling
    scales['x_scale'] = np.array([
        float(L_ref), float(L_ref), float(L_ref),      # Position
        float(V_ref), float(V_ref), float(V_ref),       # Velocity
        1.0, 1.0, 1.0, 1.0,       # Quaternion
        float(W_ref), float(W_ref), float(W_ref),       # Angular velocity
        float(M_ref)                     # Mass
    ], dtype=float)
    
    # Control scaling
    scales['u_scale'] = np.array([
        float(F_ref),                     # Thrust
        0.1745,                    # Gimbal angles (10 deg)
        0.1745,
        0.1745                     # Control surface
    ], dtype=float)
    
    scales['t_scale'] = float(T_ref)
    
    return scales


def _detect_linear_solver(solver_config):
    """
    Detect available linear solver, trying HSL first, then MUMPS.
    
    Args:
        solver_config: Solver configuration dict
    
    Returns:
        str: Name of available linear solver
    """
    import casadi as ca
    
    # Priority order: best performance first
    # Note: HSL solvers require libhsl.so to be available at runtime
    # MUMPS is often easier to install and works well
    solvers_to_try = [
        'mumps',  # Alternative: Good performance, usually easier to install
        'ma97',   # HSL: Best, parallel (requires libhsl.so)
        'ma86',   # HSL: Very good, parallel (requires libhsl.so)
        'ma77',   # HSL: Good, parallel (requires libhsl.so)
        'ma57',   # HSL: Robust, sequential (requires libhsl.so)
        'ma27',   # HSL: Basic, sequential (requires libhsl.so)
    ]
    
    # Try user preference first if set
    preferred = solver_config.get('linear_solver', None)
    if preferred and preferred.lower() != 'auto':
        solvers_to_try.insert(0, preferred.lower())
    
    for solver in solvers_to_try:
        try:
            # Quick test: try to create solver with this linear solver
            test_nlp = {'x': ca.MX.sym('x'), 'f': ca.MX.sym('x')**2}
            test_opts = {'ipopt.linear_solver': solver, 'ipopt.print_level': 0}
            # Use unique name to avoid conflicts
            test_solver = ca.nlpsol(f'test_ls_{solver}', 'ipopt', test_nlp, test_opts)
            # If we got here without exception, solver is available
            if preferred and solver != preferred.lower():
                print(f"Note: Using linear solver '{solver}' (preferred '{preferred}' not available)")
            return solver
        except RuntimeError as e:
            # Only catch RuntimeErrors (library loading issues)
            error_msg = str(e).lower()
            if "hsl" in error_msg or "libhsl" in error_msg or "library loading" in error_msg or "unknown" in error_msg:
                # This solver not available, try next
                continue
            else:
                # Other error, still try this solver (might work at solve time)
                return solver
        except Exception:
            # Unexpected error, skip this solver
            continue
    
    # Fallback: use ma27 and hope it works (may fail at solve time)
    print("Warning: Could not detect available linear solver, using 'ma27' (may fail)")
    return 'ma27'


def solve_ocp(
    params: Dict,
    limits: Dict,
    ocp_config: Dict,
    scales: Optional[Dict] = None,
    initial_guess: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
) -> Dict:
    """
    Solve optimal control problem.
    
    Args:
        params: Physical parameters
        limits: Operational limits
        ocp_config: OCP configuration
        scales: Scaling factors (optional)
        initial_guess: Initial guess (X0, U0, tf) (optional)
        
    Returns:
        result: Dictionary with solution, statistics, etc.
    """
    # Problem dimensions
    nx = ocp_config['problem']['nx']
    nu = ocp_config['problem']['nu']
    N = ocp_config['problem']['transcription']['n_intervals']
    
    # Load scales
    if scales is None:
        scales_path = Path(__file__).parent.parent.parent / 'configs' / 'scales.yaml'
        scales = load_scales(str(scales_path))
    
    # Create transcription
    transcription = DirectCollocation(
        nx=nx,
        nu=nu,
        N=N,
        params=params,
        scaling=scales
    )
    
    # Check if final time is free
    tf_fixed = ocp_config['problem']['time'].get('tf_fixed', None)
    tf_free = tf_fixed is None
    
    # Set tf_fixed in params if provided
    if tf_fixed is not None:
        params['tf_fixed'] = float(tf_fixed)
    
    # Create NLP
    objective_type = ocp_config['problem']['objective']['type']
    constraint_types = {
        'dynamic_pressure': ocp_config['path_constraints']['dynamic_pressure']['enabled'],
        'load_factor': ocp_config['path_constraints']['load_factor']['enabled'],
        'mass': ocp_config['path_constraints']['mass']['enabled']
    }
    
    nlp = transcription.create_nlp(
        objective_type=objective_type,
        constraint_types=constraint_types,
        tf_free=tf_free
    )
    
    # Setup solver
    solver_config = ocp_config['solver']
    
    # Adaptive iteration limit based on problem size
    # Larger problems (more intervals) need more iterations
    base_max_iter = int(solver_config['max_iter'])
    N = ocp_config['problem']['transcription']['n_intervals']
    # Scale iterations: N=10 → base, N=20 → 1.5x, N=30 → 2x, N=60 → 3x
    adaptive_max_iter = int(base_max_iter * (1.0 + (N - 10) * 0.05))
    adaptive_max_iter = max(base_max_iter, min(adaptive_max_iter, 5000))  # Cap at 5000
    
    opts = {
        'ipopt.max_iter': adaptive_max_iter,
        'ipopt.tol': float(solver_config['tol']),
        'ipopt.acceptable_tol': float(solver_config['acceptable_tol']),
        'ipopt.acceptable_iter': int(solver_config['acceptable_iter']),
        'ipopt.print_level': int(solver_config['print_level']),
        'ipopt.linear_solver': _detect_linear_solver(solver_config),
        'ipopt.hessian_approximation': str(solver_config['hessian_approximation']),
        'ipopt.warm_start_init_point': str(solver_config['warm_start_init_point']),
        'ipopt.mu_init': float(solver_config['mu_init']),
        'ipopt.mu_strategy': str(solver_config['mu_strategy']),
        'ipopt.max_cpu_time': float(solver_config['max_cpu_time'])
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Create bounds
    lbx_list = []
    ubx_list = []
    
    # State bounds (repeat for each node: N+1 nodes)
    lbx_state, ubx_state = create_state_bounds(nx, ocp_config.get('bounds', {}))
    lbx_list.append(np.tile(lbx_state, N + 1))
    ubx_list.append(np.tile(ubx_state, N + 1))
    
    # Control bounds (repeat for each interval: N intervals)
    lbu, ubu = create_control_bounds(nu, limits, ocp_config.get('bounds', {}))
    lbx_list.append(np.tile(lbu, N))
    ubx_list.append(np.tile(ubu, N))
    
    # Time bounds (if free)
    if tf_free:
        tf_min = ocp_config['problem']['time']['tf_min']
        tf_max = ocp_config['problem']['time']['tf_max']
        lbx_list.append([tf_min])
        ubx_list.append([tf_max])
    
    lbx = np.concatenate(lbx_list).astype(float)
    ubx = np.concatenate(ubx_list).astype(float)
    
    # Constraint bounds
    n_defects = nx * N
    n_path_nodes = N + 1
    n_path_constraints = sum(constraint_types.values())
    n_path = n_path_nodes * n_path_constraints
    
    lbg, ubg = create_constraint_bounds(n_defects, n_path, constraint_types)
    # Ensure float type
    lbg = lbg.astype(float)
    ubg = ubg.astype(float)
    
    # Initial guess
    if initial_guess is None:
        initial_conditions = ocp_config['initial']
        # Ensure params has T_max from limits for initial guess generation
        params_guess = params.copy()
        if 'T_max' not in params_guess or params_guess.get('T_max', 0) > 1e5:
            params_guess['T_max'] = limits.get('T_max', 4000.0)
        X0, U0, tf_guess = generate_initial_guess(nx, nu, N, params_guess, initial_conditions, ocp_config)
    else:
        X0, U0, tf_guess = initial_guess
    
    # Flatten initial guess (ensure float type and check for NaN/Inf)
    x0 = np.concatenate([X0.flatten(), U0.flatten()]).astype(float)
    if tf_free:
        x0 = np.concatenate([x0, [float(tf_guess)]]).astype(float)
    
    # Validate initial guess
    if np.any(~np.isfinite(x0)):
        invalid_idx = np.where(~np.isfinite(x0))[0]
        print(f"WARNING: Initial guess contains invalid values at {len(invalid_idx)} indices")
        if len(invalid_idx) <= 10:
            print(f"  Indices: {invalid_idx}")
            print(f"  Values: {x0[invalid_idx]}")
        # Replace with reasonable defaults
        x0[~np.isfinite(x0)] = 0.0
    
    # Fix quaternion normalization and ensure within bounds
    nx = X0.shape[0]
    for k in range(N + 1):
        q_start = k * nx + 6
        q_end = q_start + 4
        if q_end <= len(x0):
            q = x0[q_start:q_end]
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-10 or not np.isfinite(q_norm):
                x0[q_start:q_end] = [1.0, 0.0, 0.0, 0.0]
            else:
                # Normalize
                x0[q_start:q_end] = q / q_norm
    
    # Clamp initial guess to bounds
    x0 = np.clip(x0, lbx, ubx)
    
    # Validate initial guess one more time before solve
    if np.any(~np.isfinite(x0)):
        print(f"ERROR: Initial guess still contains invalid values after validation!")
        invalid_count = np.sum(~np.isfinite(x0))
        print(f"  Invalid values: {invalid_count} out of {len(x0)}")
        # Try to fix again
        x0[~np.isfinite(x0)] = 0.0
    
    # Check bounds compatibility
    bound_violations = np.sum((x0 < lbx) | (x0 > ubx))
    if bound_violations > 0:
        print(f"WARNING: {bound_violations} initial guess values violate bounds, clamping...")
        x0 = np.clip(x0, lbx, ubx)
    
    # Solve
    print("Solving OCP...")
    try:
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    except RuntimeError as e:
        error_msg = str(e)
        if "Invalid number" in error_msg or "NaN" in error_msg or "Inf" in error_msg:
            print(f"\nERROR: Invalid numbers detected during solve.")
            print(f"This usually means:")
            print(f"  1. Initial guess produces NaN in dynamics evaluation")
            print(f"  2. Constraint evaluation fails at initial point")
            print(f"  3. Scaling issues")
            print(f"\nTry:")
            print(f"  - Reduce problem size (n_intervals)")
            print(f"  - Disable path constraints")
            print(f"  - Check initial guess generation")
            raise RuntimeError(f"IPOPT detected invalid numbers: {error_msg}")
        raise
    
    # Extract solution
    x_opt = sol['x']
    
    # Convert to numpy and reshape solution
    x_opt_np = np.array(x_opt)
    
    # Reshape solution
    X_opt = x_opt_np[:nx * (N + 1)].reshape(nx, N + 1)
    U_opt = x_opt_np[nx * (N + 1):nx * (N + 1) + nu * N].reshape(nu, N)
    if tf_free:
        tf_opt = float(x_opt[-1])
    else:
        tf_opt = tf_guess
    
    # Time grid
    t = np.linspace(0.0, tf_opt, N + 1)
    
    # Statistics
    solver_stats = solver.stats()
    success = solver_stats['success']
    constraint_violation = float(np.max(np.abs(sol['g'])))  # Absolute value for violation
    
    stats = {
        'success': success,
        'converged': success,  # Alias for consistency
        'iterations': solver_stats['iter_count'],
        'solver_time': solver_stats.get('t_wall_total', 0.0),
        'objective': float(sol['f']),
        'constraint_violation': constraint_violation,
        'status': solver_stats.get('return_status', 'unknown')
    }
    
    result = {
        'X': X_opt,
        'U': U_opt,
        't': t,
        'tf': tf_opt,
        'stats': stats,
        'config': ocp_config,
        'params': params,
        'limits': limits
    }
    
    return result


def save_results(result: Dict, output_path: str):
    """Save results to HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X', data=result['X'])
        f.create_dataset('U', data=result['U'])
        f.create_dataset('t', data=result['t'])
        f.attrs['tf'] = result['tf']
        f.attrs['objective'] = result['stats']['objective']
        f.attrs['iterations'] = result['stats']['iterations']
        f.attrs['solver_time'] = result['stats']['solver_time']
        f.attrs['success'] = result['stats']['success']
        f.attrs['timestamp'] = datetime.now().isoformat()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve rocket trajectory OCP')
    parser.add_argument('--phys', type=str, default='configs/phys.yaml',
                       help='Physical parameters file')
    parser.add_argument('--limits', type=str, default='configs/limits.yaml',
                       help='Limits file')
    parser.add_argument('--ocp', type=str, default='configs/ocp.yaml',
                       help='OCP configuration file')
    parser.add_argument('--scales', type=str, default='configs/scales.yaml',
                       help='Scales file')
    parser.add_argument('--output', type=str, default='data/raw/ocp_runs/run_0001.h5',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load configurations
    base_path = Path(__file__).parent.parent.parent
    params, limits, ocp_config = load_parameters(
        str(base_path / args.phys),
        str(base_path / args.limits),
        str(base_path / args.ocp)
    )
    scales = load_scales(str(base_path / args.scales))
    
    # Solve
    result = solve_ocp(params, limits, ocp_config, scales)
    
    # Save results
    output_path = base_path / args.output
    save_results(result, str(output_path))
    
    print(f"\nOCP solved successfully!")
    print(f"  Objective: {result['stats']['objective']:.6f}")
    print(f"  Iterations: {result['stats']['iterations']}")
    print(f"  Solver time: {result['stats']['solver_time']:.2f} s")
    print(f"  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

