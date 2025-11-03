"""
Utility functions for optimal control problem setup.

Includes initial guess generation and helper functions.
"""

import numpy as np
import casadi as ca
from typing import Dict, Tuple, Optional
import h5py


def generate_vertical_ascent_guess(
    nx: int,
    nu: int,
    N: int,
    params: Dict,
    initial_conditions: Dict,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate initial guess using simple vertical ascent.
    
    Args:
        nx: State dimension
        nu: Control dimension
        N: Number of intervals
        params: Physical parameters
        initial_conditions: Initial state dictionary
        config: Configuration dictionary for guess parameters
        
    Returns:
        X0: Initial state guess (nx, N+1)
        U0: Initial control guess (nu, N)
        tf_guess: Initial time guess [s]
    """
    config = config or {}
    guess_config = config.get('vertical_ascent', {})
    
    # Time guess
    tf_min = config.get('tf_min', 30.0)
    tf_max = config.get('tf_max', 120.0)
    tf_guess = config.get('tf_fixed', 0.5 * (tf_min + tf_max))
    
    # Thrust parameters (get from params or limits, use reasonable default)
    T_max = params.get('T_max', None)
    if T_max is None:
        # Try to get from limits if passed
        T_max = params.get('limits', {}).get('T_max', 4000.0)
    if T_max is None or T_max > 1e6:
        # Sanity check: use reasonable default
        T_max = 4000.0
    T_initial = guess_config.get('T_initial', 0.8 * T_max)
    T_taper_time = guess_config.get('T_taper_time', 0.8 * tf_guess)
    
    # Gimbal angles
    theta_g_initial = guess_config.get('theta_g_initial', np.pi / 2.0)  # 90 deg
    theta_g_final = guess_config.get('theta_g_final', 0.0)
    phi_g = guess_config.get('phi_g', 0.0)
    
    # Time grid
    t = np.linspace(0.0, tf_guess, N + 1)
    dt = tf_guess / N
    
    # Initialize arrays
    X0 = np.zeros((nx, N + 1))
    U0 = np.zeros((nu, N))
    
    # Initial state
    X0[0, 0] = initial_conditions.get('x', 0.0)  # x
    X0[1, 0] = initial_conditions.get('y', 0.0)  # y
    X0[2, 0] = initial_conditions.get('z', 0.0)  # z
    X0[3, 0] = initial_conditions.get('vx', 0.0)  # vx
    X0[4, 0] = initial_conditions.get('vy', 0.0)  # vy
    X0[5, 0] = initial_conditions.get('vz', 0.0)  # vz
    X0[6, 0] = initial_conditions.get('q0', 1.0)  # q0
    X0[7, 0] = initial_conditions.get('q1', 0.0)  # q1
    X0[8, 0] = initial_conditions.get('q2', 0.0)  # q2
    X0[9, 0] = initial_conditions.get('q3', 0.0)  # q3
    X0[10, 0] = initial_conditions.get('wx', 0.0)  # wx
    X0[11, 0] = initial_conditions.get('wy', 0.0)  # wy
    X0[12, 0] = initial_conditions.get('wz', 0.0)  # wz
    m0 = initial_conditions.get('m', params.get('m0', 5000.0))
    X0[13, 0] = m0
    
    # Integrate forward using simple dynamics
    g0 = params.get('g0', 9.81)
    Isp = params.get('Isp', 300.0)
    m_dry = params.get('m_dry', 1000.0)
    
    for k in range(N):
        # Current state
        r = X0[0:3, k]
        v = X0[3:6, k]
        q = X0[6:10, k]
        w = X0[10:13, k]
        m = X0[13, k]
        
        # Control
        t_k = t[k]
        
        # Thrust (taper after T_taper_time)
        if t_k < T_taper_time:
            T = T_initial
        else:
            T = T_initial * (1.0 - (t_k - T_taper_time) / (tf_guess - T_taper_time))
        T = max(T, 0.0)
        U0[0, k] = T
        
        # Gimbal angles (taper from vertical to horizontal)
        if t_k < T_taper_time:
            theta_g = theta_g_initial + (theta_g_final - theta_g_initial) * (t_k / T_taper_time)
        else:
            theta_g = theta_g_final
        U0[1, k] = theta_g
        U0[2, k] = phi_g
        U0[3, k] = 0.0  # No control surface deflection
        
        # Simple vertical dynamics
        # Position update
        r_next = r + v * dt
        
        # Velocity update (vertical ascent with constant acceleration)
        a = (T / m) - g0
        v_next = v + np.array([0.0, 0.0, a]) * dt
        v_next[0] = 0.0  # No horizontal velocity
        v_next[1] = 0.0
        
        # Mass update
        m_dot = -T / (Isp * g0)
        m_next = m + m_dot * dt
        m_next = max(m_next, m_dry)
        
        # Quaternion (keep identity for vertical)
        q_next = q.copy()
        q_norm = np.linalg.norm(q_next)
        if q_norm > 1e-10:
            q_next = q_next / q_norm
        else:
            q_next = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Angular velocity (zero)
        w_next = w.copy()
        
        # Store next state
        X0[0:3, k + 1] = r_next
        X0[3:6, k + 1] = v_next
        X0[6:10, k + 1] = q_next
        X0[10:13, k + 1] = w_next
        X0[13, k + 1] = m_next
    
    return X0, U0, tf_guess


def generate_polynomial_guess(
    nx: int,
    nu: int,
    N: int,
    params: Dict,
    initial_conditions: Dict,
    terminal_conditions: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate initial guess using polynomial interpolation.
    
    Args:
        nx: State dimension
        nu: Control dimension
        N: Number of intervals
        params: Physical parameters
        initial_conditions: Initial state dictionary
        terminal_conditions: Terminal state dictionary (optional)
        config: Configuration dictionary
        
    Returns:
        X0: Initial state guess
        U0: Initial control guess
        tf_guess: Initial time guess
    """
    config = config or {}
    tf_guess = config.get('tf_fixed', 100.0)
    
    # Time grid
    t = np.linspace(0.0, tf_guess, N + 1)
    tau = t / tf_guess  # Normalized time [0, 1]
    
    # Initialize
    X0 = np.zeros((nx, N + 1))
    U0 = np.zeros((nu, N))
    
    # Initial state
    X0[:, 0] = np.array([
        initial_conditions.get('x', 0.0),
        initial_conditions.get('y', 0.0),
        initial_conditions.get('z', 0.0),
        initial_conditions.get('vx', 0.0),
        initial_conditions.get('vy', 0.0),
        initial_conditions.get('vz', 0.0),
        initial_conditions.get('q0', 1.0),
        initial_conditions.get('q1', 0.0),
        initial_conditions.get('q2', 0.0),
        initial_conditions.get('q3', 0.0),
        initial_conditions.get('wx', 0.0),
        initial_conditions.get('wy', 0.0),
        initial_conditions.get('wz', 0.0),
        initial_conditions.get('m', params.get('m0', 5000.0))
    ])
    
    # Terminal state (if provided, else extrapolate)
    if terminal_conditions:
        xf = np.array([
            terminal_conditions.get('x', X0[0, 0]),
            terminal_conditions.get('y', X0[1, 0]),
            terminal_conditions.get('z', X0[2, 0] + 10000.0),
            terminal_conditions.get('vx', X0[3, 0]),
            terminal_conditions.get('vy', X0[4, 0]),
            terminal_conditions.get('vz', X0[5, 0] + 100.0),
            1.0, 0.0, 0.0, 0.0,  # Identity quaternion
            0.0, 0.0, 0.0,
            params.get('m_dry', 1000.0)
        ])
    else:
        # Simple extrapolation
        xf = X0[:, 0].copy()
        xf[2] += 10000.0  # Altitude
        xf[5] += 100.0    # Vertical velocity
        xf[13] = params.get('m_dry', 1000.0)  # Dry mass
    
    # Polynomial interpolation for each state
    for i in range(nx):
        if i in [6, 7, 8, 9]:  # Quaternion - keep normalized
            if i == 6:  # q0
                X0[i, :] = 1.0 - tau  # Linear interpolation
            else:
                X0[i, :] = 0.0
        else:
            # Linear interpolation
            X0[i, :] = X0[i, 0] + (xf[i] - X0[i, 0]) * tau
    
    # Normalize quaternions (handle zero norm case)
    for k in range(N + 1):
        q = X0[6:10, k]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
        else:
            # Default to identity quaternion if norm is too small
            q = np.array([1.0, 0.0, 0.0, 0.0])
        X0[6:10, k] = q
    
    # Control guess (taper thrust, zero gimbal)
    T_max = params.get('T_max', None)
    if T_max is None:
        T_max = params.get('limits', {}).get('T_max', 4000.0)
    if T_max is None or T_max > 1e6:
        T_max = 4000.0  # Reasonable default
    T0 = 0.8 * T_max
    for k in range(N):
        U0[0, k] = T0 * (1.0 - tau[k])  # Taper to zero
        U0[1, k] = (np.pi / 2.0) * (1.0 - tau[k])  # Taper gimbal
        U0[2, k] = 0.0
        U0[3, k] = 0.0
    
    return X0, U0, tf_guess


def load_initial_guess_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load initial guess from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        X0: Initial state guess
        U0: Initial control guess
        tf_guess: Initial time guess
    """
    with h5py.File(filepath, 'r') as f:
        X0 = f['X'][:]
        U0 = f['U'][:]
        tf_guess = f.attrs.get('tf', 100.0)
    
    return X0, U0, tf_guess


def generate_initial_guess(
    nx: int,
    nu: int,
    N: int,
    params: Dict,
    initial_conditions: Dict,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate initial guess based on strategy.
    
    Args:
        nx: State dimension
        nu: Control dimension
        N: Number of intervals
        params: Physical parameters
        initial_conditions: Initial state dictionary
        config: Configuration dictionary
        
    Returns:
        X0: Initial state guess
        U0: Initial control guess
        tf_guess: Initial time guess
    """
    strategy = config.get('initial_guess', {}).get('strategy', 'vertical_ascent')
    
    if strategy == 'vertical_ascent':
        return generate_vertical_ascent_guess(nx, nu, N, params, initial_conditions, config)
    elif strategy == 'polynomial':
        return generate_polynomial_guess(nx, nu, N, params, initial_conditions, None, config)
    elif strategy == 'from_file':
        filepath = config.get('initial_guess', {}).get('file', None)
        if filepath is None:
            raise ValueError("Strategy is 'from_file' but no file path provided")
        return load_initial_guess_from_file(filepath)
    else:
        # Default to vertical ascent
        return generate_vertical_ascent_guess(nx, nu, N, params, initial_conditions, config)


def scale_states(X: np.ndarray, x_scale: np.ndarray) -> np.ndarray:
    """Scale states using reference scales."""
    return X / x_scale[:, np.newaxis]


def unscale_states(X_scaled: np.ndarray, x_scale: np.ndarray) -> np.ndarray:
    """Unscale states using reference scales."""
    return X_scaled * x_scale[:, np.newaxis]


def scale_controls(U: np.ndarray, u_scale: np.ndarray) -> np.ndarray:
    """Scale controls using reference scales."""
    return U / u_scale[:, np.newaxis]


def unscale_controls(U_scaled: np.ndarray, u_scale: np.ndarray) -> np.ndarray:
    """Unscale controls using reference scales."""
    return U_scaled * u_scale[:, np.newaxis]

