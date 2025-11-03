"""
Constraint handling for optimal control problem.

Computes constraint bounds and provides helper functions for constraint evaluation.
"""

import casadi as ca
import numpy as np
from typing import Dict, Tuple, Optional
from .dynamics_casadi import compute_dynamic_pressure, compute_load_factor


def compute_dynamic_pressure_constraint(
    x: ca.MX,
    params: Dict,
    q_max: Optional[float] = None
) -> ca.MX:
    """
    Compute dynamic pressure constraint: q <= q_max.
    
    Args:
        x: State vector or state matrix
        params: Physical parameters
        q_max: Maximum dynamic pressure (if None, use params['q_max'])
        
    Returns:
        constraint: q - q_max (should be <= 0)
    """
    q = compute_dynamic_pressure(x, params)
    q_max_val = q_max if q_max is not None else params.get('q_max', 50000.0)
    return q - q_max_val


def compute_load_factor_constraint(
    x: ca.MX,
    u: ca.MX,
    params: Dict,
    n_max: Optional[float] = None
) -> ca.MX:
    """
    Compute load factor constraint: n <= n_max.
    
    Args:
        x: State vector or state matrix
        u: Control vector or control matrix
        params: Physical parameters
        n_max: Maximum load factor (if None, use params['n_max'])
        
    Returns:
        constraint: n - n_max (should be <= 0)
    """
    n = compute_load_factor(x, u, params)
    n_max_val = n_max if n_max is not None else params.get('n_max', 10.0)
    return n - n_max_val


def compute_mass_constraint(
    x: ca.MX,
    m_dry: Optional[float] = None,
    params: Optional[Dict] = None
) -> ca.MX:
    """
    Compute mass constraint: m >= m_dry.
    
    Args:
        x: State vector or state matrix
        m_dry: Dry mass (if None, use params['m_dry'])
        params: Physical parameters (optional)
        
    Returns:
        constraint: m_dry - m (should be <= 0)
    """
    if len(x.shape) == 1:
        m = x[13]
    else:
        m = x[13, :]
    
    m_dry_val = m_dry
    if m_dry_val is None and params is not None:
        m_dry_val = params.get('m_dry', 1000.0)
    elif m_dry_val is None:
        m_dry_val = 1000.0
    
    return m_dry_val - m


def compute_quaternion_norm_constraint(x: ca.MX) -> ca.MX:
    """
    Compute quaternion norm constraint: ||q||^2 = 1.
    
    Args:
        x: State vector or state matrix
        
    Returns:
        constraint: ||q||^2 - 1 (should be = 0)
    """
    if len(x.shape) == 1:
        q = x[6:10]
    else:
        q = x[6:10, :]
    
    q_norm_sq = ca.sum1(q**2)
    return q_norm_sq - 1.0


def create_state_bounds(
    nx: int,
    bounds_config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create state bounds.
    
    Args:
        nx: State dimension
        bounds_config: Configuration dictionary with 'x_min' and 'x_max'
        
    Returns:
        lbx: Lower bounds (nx,)
        ubx: Upper bounds (nx,)
    """
    if bounds_config is None:
        # Default bounds
        lbx = np.array([
            -1e6, -1e6, -100.0,        # x, y, z
            -1e4, -1e4, -1e4,          # vx, vy, vz
            -1.1, -1.1, -1.1, -1.1,   # q0, q1, q2, q3
            -10.0, -10.0, -10.0,       # wx, wy, wz
            0.0                        # m
        ])
        ubx = np.array([
            1e6, 1e6, 1e6,
            1e4, 1e4, 1e4,
            1.1, 1.1, 1.1, 1.1,
            10.0, 10.0, 10.0,
            1e6
        ])
    else:
        lbx = np.array(bounds_config.get('x_min', [-1e6] * nx))
        ubx = np.array(bounds_config.get('x_max', [1e6] * nx))
    
    return lbx, ubx


def create_control_bounds(
    nu: int,
    limits: Dict,
    bounds_config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create control bounds.
    
    Args:
        nu: Control dimension
        limits: Limits dictionary with T_max, theta_max, etc.
        bounds_config: Configuration dictionary with 'u_min' and 'u_max'
        
    Returns:
        lbu: Lower bounds (nu,)
        ubu: Upper bounds (nu,)
    """
    T_max = limits.get('T_max', 1000000.0)
    theta_max = limits.get('theta_max', 0.1745)  # ~10 degrees
    phi_max = limits.get('phi_max', 0.1745)
    delta_max = limits.get('delta_max', 0.1745)
    
    if bounds_config is None:
        lbu = np.array([0.0, -theta_max, -phi_max, -delta_max])
        ubu = np.array([T_max, theta_max, phi_max, delta_max])
    else:
        lbu = np.array(bounds_config.get('u_min', [0.0, -theta_max, -phi_max, -delta_max]))
        ubu = np.array(bounds_config.get('u_max', [T_max, theta_max, phi_max, delta_max]))
        
        # Replace None with actual limits
        if bounds_config.get('u_max') is not None:
            ubu_list = list(ubu)
            for i, val in enumerate(bounds_config['u_max']):
                if val is None:
                    if i == 0:
                        ubu_list[i] = T_max
                    else:
                        ubu_list[i] = [theta_max, phi_max, delta_max][i - 1]
            ubu = np.array(ubu_list)
    
    return lbu, ubu


def create_constraint_bounds(
    n_defects: int,
    n_path: int,
    constraint_types: Dict[str, bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create constraint bounds for NLP.
    
    Args:
        n_defects: Number of defect constraints (nx * N)
        n_path: Number of path constraints per node
        constraint_types: Dictionary of constraint flags
        
    Returns:
        lbg: Lower bounds for constraints
        ubg: Upper bounds for constraints
    """
    # Defect constraints: should be zero
    lbg_defect = np.zeros(n_defects)
    ubg_defect = np.zeros(n_defects)
    
    # Path constraints
    # n_path = n_nodes * n_constraints, so n_nodes = n_path / n_constraints
    n_constraints = sum(constraint_types.values())
    n_nodes = n_path // n_constraints if n_constraints > 0 and n_path > 0 else 0
    
    lbg_path = []
    ubg_path = []
    
    for k in range(n_nodes):
        # Dynamic pressure: q <= q_max
        if constraint_types.get('dynamic_pressure', False):
            lbg_path.append(-np.inf)
            ubg_path.append(0.0)
        
        # Load factor: n <= n_max
        if constraint_types.get('load_factor', False):
            lbg_path.append(-np.inf)
            ubg_path.append(0.0)
        
        # Mass: m >= m_dry, so m_dry - m <= 0
        if constraint_types.get('mass', False):
            lbg_path.append(-np.inf)
            ubg_path.append(0.0)
    
    lbg = np.concatenate([lbg_defect, lbg_path])
    ubg = np.concatenate([ubg_defect, ubg_path])
    
    return lbg, ubg

